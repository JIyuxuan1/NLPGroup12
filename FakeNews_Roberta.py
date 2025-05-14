import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset, Dataset, Features, ClassLabel, Value
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
import numpy as np
import wandb
import os
import time
from tqdm.auto import tqdm
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", message=".*resume_download.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Using `TRANSFORMERS_CACHE`.*")


# --- Configuration ---
config = {
    "model_name": "roberta-base",
    "dataset_path": "train.csv",
    "label_column": "label",
    "title_column": "title",
    "text_column": "text",
    "output_dir": "./roberta-base_fake_news_checkpoints",
    "num_epochs": 5,
    "batch_size": 32,
    "learning_rate": 1e-5,
    "warmup_steps": 100,
    "max_seq_length": 512,  # RoBERTa's max sequence length
    "test_size": 0.2,
    "seed": 42,
    "wandb_project": "fake-news-classification-roberta",
    "label_names": ["Fake", "Real"] # Corresponds to 0 and 1
}

# --- Helper Functions ---

def preprocess_data(examples, tokenizer):
    """Combines title and text, then tokenizes."""
    # Handle potential missing values gracefully
    titles = [str(t) if t else "" for t in examples[config['title_column']]]
    texts = [str(t) if t else "" for t in examples[config['text_column']]]

    # Combine title and text with a separator token
    # Using tokenizer.sep_token ensures compatibility
    full_texts = [
        title + f" {tokenizer.sep_token} " + text
        for title, text in zip(titles, texts)
    ]
    # Tokenize
    tokenized = tokenizer(
        full_texts,
        truncation=True,
        padding=False, # Let DataCollator handle padding
        max_length=config['max_seq_length']
    )
    return tokenized

def compute_metrics(predictions, labels, label_names):
    """Computes evaluation metrics."""
    acc = accuracy_score(labels, predictions)
    # Calculate precision, recall, f1 for each class and macro/weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    report = classification_report(
        labels, predictions, target_names=label_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(labels, predictions)

    return {
        "accuracy": acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm
    }

def evaluate_model(model, dataloader, device, label_names):
    """Evaluates the model on a given dataloader."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            labels = batch.pop("labels").to(device)
            # Move all input tensors to the device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1) # Get probabilities for PR curve

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_preds, all_labels, label_names)
    metrics["eval_loss"] = avg_loss

    # Ensure probabilities are in the right format for wandb.plot.pr_curve
    # It expects probabilities for each class (num_samples, num_classes)
    all_probs_np = np.array(all_probs)

    return metrics, all_labels, all_preds, all_probs_np


# --- Main Script ---

if __name__ == "__main__":
    start_time = time.time()

    # 1. Initialize W&B
    wandb.login() # Assumes WANDB_API_KEY is set or prompts login
    run = wandb.init(
        project=config['wandb_project'],
        config=config,
        name=f"{config['model_name']}-run-{int(time.time())}" # Unique run name
    )

    # Set seed for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])

    # 2. Load and Preprocess Data
    print("Loading and preprocessing data...")
    # Define features including ClassLabel for stratification
    features = Features({
        config['title_column']: Value('string'),
        config['text_column']: Value('string'),
        config['label_column']: ClassLabel(num_classes=2, names=config['label_names'])
    })
    # Load dataset using the defined features
    raw_dataset = load_dataset('csv', data_files=config['dataset_path'], features=features, split='train')

    # Initialize tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained(config['model_name'])

    # Tokenize and combine text fields
    tokenized_dataset = raw_dataset.map(
        lambda examples: preprocess_data(examples, tokenizer),
        batched=True,
        remove_columns=[config['title_column'], config['text_column']] # Remove original text columns
    )

    # Split dataset (stratified by label)
    processed_datasets = tokenized_dataset.train_test_split(
        test_size=config['test_size'],
        seed=config['seed'],
        stratify_by_column=config['label_column'] # Crucial for balanced splits
    )
    train_dataset = processed_datasets['train']
    test_dataset = processed_datasets['test']

    # Format datasets for PyTorch
    train_dataset.set_format("torch", columns=['input_ids', 'attention_mask', config['label_column']])
    test_dataset.set_format("torch", columns=['input_ids', 'attention_mask', config['label_column']])
    # Rename label column for compatibility with Hugging Face models
    train_dataset = train_dataset.rename_column(config['label_column'], "labels")
    test_dataset = test_dataset.rename_column(config['label_column'], "labels")


    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # 3. Model, Optimizer, Scheduler, DataLoaders
    print("Setting up model, optimizer, scheduler, and dataloaders...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = RobertaForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=len(config['label_names']) # Binary classification
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False, # No need to shuffle test data
        collate_fn=data_collator
    )

    num_training_steps = config['num_epochs'] * len(train_dataloader)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=num_training_steps
    )

    # 4. Initial Evaluation (Before Training)
    print("Performing initial evaluation before training...")
    initial_metrics, _, _, _ = evaluate_model(model, test_dataloader, device, config['label_names'])
    print(f"Initial Test Accuracy: {initial_metrics['accuracy']:.4f}")
    wandb.log({"initial_test_metrics": initial_metrics}, step=0) # Log initial metrics at step 0


    # 5. Training Loop
    print("Starting training...")
    os.makedirs(config['output_dir'], exist_ok=True)
    best_val_accuracy = 0.0
    global_step = 0

    for epoch in range(config['num_epochs']):
        epoch_start_time = time.time()
        print(f"\n--- Epoch {epoch + 1}/{config['num_epochs']} ---")

        # --- Training Phase ---
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training", leave=False)

        for batch in train_progress_bar:
            optimizer.zero_grad()
            labels = batch.pop("labels").to(device)
            # Move all input tensors to the device
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch, labels=labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 防止提督爆炸
            optimizer.step()
            lr_scheduler.step()

            # Log training loss per step
            wandb.log({"train_loss_step": loss.item()}, step=global_step)
            train_progress_bar.set_postfix({'loss': loss.item()})
            global_step += 1

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} Training Loss: {avg_train_loss:.4f}")

        # --- Evaluation Phase ---
        print(f"Evaluating on test set for Epoch {epoch + 1}...")
        eval_metrics, eval_labels, eval_preds, eval_probs = evaluate_model(model, test_dataloader, device, config['label_names'])

        # Prepare metrics for logging (remove complex objects like numpy arrays for general logging)
        log_metrics = {k: v for k, v in eval_metrics.items() if k not in ["classification_report", "confusion_matrix"]}
        log_metrics["avg_train_loss"] = avg_train_loss # Log avg train loss for the epoch

        print(f"Epoch {epoch + 1} Test Accuracy: {eval_metrics['accuracy']:.4f}")
        print(f"Epoch {epoch + 1} Test F1 Macro: {eval_metrics['f1_macro']:.4f}")
        print(f"Epoch {epoch + 1} Test F1 Weighted: {eval_metrics['f1_weighted']:.4f}")

        # Log metrics to W&B
        wandb.log({**log_metrics, "epoch": epoch + 1}, step=global_step)

        # --- Model Checkpointing ---
        current_accuracy = eval_metrics['accuracy']
        epoch_output_dir = os.path.join(config['output_dir'], f"epoch_{epoch+1}")
        os.makedirs(epoch_output_dir, exist_ok=True)

        print(f"Saving model checkpoint for epoch {epoch + 1} to {epoch_output_dir}")
        model.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)

        # if current_accuracy > best_val_accuracy:
        #     print(f"Validation accuracy improved ({best_val_accuracy:.4f} -> {current_accuracy:.4f}). Saving best model.")
        #     best_val_accuracy = current_accuracy
        #     best_model_output_dir = os.path.join(config['output_dir'], "best_model")
        #     os.makedirs(best_model_output_dir, exist_ok=True)
        #     model.save_pretrained(best_model_output_dir)
        #     tokenizer.save_pretrained(best_model_output_dir)
        #     wandb.summary["best_val_accuracy_epoch"] = epoch + 1

        epoch_end_time = time.time()
        print(f"Epoch {epoch + 1} duration: {epoch_end_time - epoch_start_time:.2f} seconds")


    # --- Final Evaluation & Logging ---
    total_training_time = time.time() - start_time
    print("\nTraining finished.")
    print(f"Total Training Time: {total_training_time:.2f} seconds")

    # Evaluate the final model (from the last epoch)
    print("Performing final evaluation on the test set...")
    final_metrics, final_labels, final_preds, final_probs = evaluate_model(model, test_dataloader, device, config['label_names'])

    # Log final summary stats to W&B
    wandb.summary["final_test_accuracy"] = final_metrics['accuracy']
    wandb.summary["final_test_f1_macro"] = final_metrics['f1_macro']
    wandb.summary["final_test_f1_weighted"] = final_metrics['f1_weighted']
    wandb.summary["total_training_time_seconds"] = total_training_time
    # wandb.summary["best_val_accuracy"] = best_val_accuracy # Uncomment if tracking best model

    # Log final detailed classification report to W&B
    # Convert report dict to DataFrame for better table logging
    report_df = pd.DataFrame(final_metrics['classification_report']).transpose()
    report_df = report_df.reset_index().rename(columns={'index': 'class'})
    wandb.log({"final_test_classification_report": wandb.Table(dataframe=report_df)})

    # Log final confusion matrix
    wandb.log({"final_test_conf_matrix": wandb.plot.confusion_matrix(
        preds=final_preds,
        y_true=final_labels,
        class_names=config['label_names']
    )})

    # Log final Precision-Recall curve
    wandb.log({"final_test_pr_curve": wandb.plot.pr_curve(
        y_true=final_labels,
        y_probas=final_probs,
        labels=config['label_names']
    )})

    # Print final classification report to console
    print("\nFinal Classification Report (Test Set):")
    print(classification_report(
        final_labels, final_preds, target_names=config['label_names'], zero_division=0
    ))

    # 6. Finish W&B Run
    run.finish()
    print("\nW&B run finished. Check the dashboard for detailed logs and visualizations.")