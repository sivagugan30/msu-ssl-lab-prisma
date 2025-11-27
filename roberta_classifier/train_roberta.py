"""
Fine-tune RoBERTa for stuttering variability classification.
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# Paths
BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "train.jsonl"
VAL_PATH = BASE_DIR / "val.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "roberta_var"

# Model
MODEL_NAME = "roberta-base"
MAX_LENGTH = 512


def load_jsonl(path: Path):
    """Load JSONL file."""
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def compute_metrics(eval_pred):
    """Compute classification metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def main():
    print("=" * 60)
    print("Training RoBERTa for Stuttering Variability Classification")
    print("=" * 60)
    
    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS device (Apple Silicon)")
    else:
        device = "cpu"
    
    # Load data
    print("\nLoading data...")
    train_data = load_jsonl(TRAIN_PATH)
    val_data = load_jsonl(VAL_PATH)
    print(f"Train samples: {len(train_data)}")
    print(f"Val samples: {len(val_data)}")
    
    # Create datasets
    train_dataset = Dataset.from_list([{"text": d["text"], "label": d["label"]} for d in train_data])
    val_dataset = Dataset.from_list([{"text": d["text"], "label": d["label"]} for d in val_data])
    
    print(f"\nTrain dataset: {train_dataset}")
    print(f"Val dataset: {val_dataset}")
    
    # Load tokenizer and model
    print(f"\nLoading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )
    
    print("\nTokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        use_mps_device=(device == "mps"),
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Train batch size: {training_args.per_device_train_batch_size}")
    print(f"  Epochs: {training_args.num_train_epochs}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final evaluation...")
    print("=" * 60)
    
    results = trainer.evaluate()
    print("\nValidation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save model
    print(f"\nSaving model to {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    print("\nTraining complete!")
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
