"""
Run multiple fine-tuning experiments with different methods and compare results.
Tracks training time, metrics, and parameter counts for each method.
"""

import json
import time
import torch
import gc
from pathlib import Path
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType, IA3Config
from trl import SFTTrainer, SFTConfig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from config import (
    MODEL_ID, TRAIN_JSONL, VAL_JSONL, OUTPUT_DIR, RESULTS_DIR, DEVICE,
    SYSTEM_PROMPT, format_training_example, format_prompt, TRAINING_CONFIG
)


# =============================================================================
# EXPERIMENT CONFIGURATIONS
# =============================================================================

EXPERIMENTS = {
    "lora_r16": {
        "name": "LoRA (r=16)",
        "method": "lora",
        "config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "num_epochs": 3,
        }
    },
    "lora_r64": {
        "name": "LoRA (r=64, High Rank)",
        "method": "lora",
        "config": {
            "r": 64,
            "lora_alpha": 128,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "num_epochs": 3,
        }
    },
    "lora_fast": {
        "name": "LoRA (r=16, High LR)",
        "method": "lora",
        "config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "learning_rate": 5e-4,
            "num_epochs": 2,
        }
    },
    "ia3": {
        "name": "IA3 (Lightweight)",
        "method": "ia3",
        "config": {
            "learning_rate": 3e-4,
            "num_epochs": 3,
        }
    },
}


def load_data(jsonl_path: Path):
    """Load data from JSONL file."""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            data.append(entry)
    return data


def prepare_dataset(data, tokenizer):
    """Convert data to the format expected by SFTTrainer."""
    formatted_data = []
    
    for entry in data:
        prompt, response = format_training_example(entry["text"], entry["label"])
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        formatted_data.append({"text": text})
    
    return Dataset.from_list(formatted_data)


def create_model_and_tokenizer():
    """Create base model and tokenizer."""
    if DEVICE == "mps":
        dtype = torch.float16
    elif DEVICE == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    
    return model, tokenizer


def apply_peft_method(model, method: str, config: dict):
    """Apply the specified PEFT method to the model."""
    
    if method == "lora":
        peft_config = LoraConfig(
            r=config["r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        )
    elif method == "ia3":
        peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=["k_proj", "v_proj", "down_proj"],
            feedforward_modules=["down_proj"],
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    model = get_peft_model(model, peft_config)
    return model, peft_config


def classify_with_logits(model, tokenizer, text: str) -> int:
    """Classify using logit comparison."""
    prompt = format_prompt(text)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        formatted,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
    
    accounted_tokens = ["accounted", "Accounted", " accounted", " Accounted"]
    not_tokens = ["not", "Not", " not", " Not", "no", "No"]
    
    accounted_probs = []
    for token in accounted_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            accounted_probs.append(next_token_logits[token_ids[0]].item())
    
    not_probs = []
    for token in not_tokens:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        if token_ids:
            not_probs.append(next_token_logits[token_ids[0]].item())
    
    max_accounted = max(accounted_probs) if accounted_probs else float('-inf')
    max_not = max(not_probs) if not_probs else float('-inf')
    
    return 1 if max_accounted > max_not else 0


def evaluate_model(model, tokenizer, val_data):
    """Evaluate model on validation set."""
    predictions = []
    labels = []
    
    model.eval()
    for entry in tqdm(val_data, desc="Evaluating", leave=False):
        pred = classify_with_logits(model, tokenizer, entry["text"])
        predictions.append(pred)
        labels.append(entry["label"])
    
    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }
    
    return metrics


def run_experiment(exp_name: str, exp_config: dict, train_data, val_data):
    """Run a single experiment."""
    print(f"\n{'='*60}")
    print(f"🔬 EXPERIMENT: {exp_config['name']}")
    print(f"{'='*60}")
    
    config = exp_config["config"]
    method = exp_config["method"]
    
    # Create fresh model
    print("Loading model...")
    model, tokenizer = create_model_and_tokenizer()
    
    # Apply PEFT method
    print(f"Applying {method.upper()}...")
    model, peft_config = apply_peft_method(model, method, config)
    
    # Get trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = trainable_params / total_params * 100
    
    print(f"Trainable params: {trainable_params:,} / {total_params:,} ({trainable_pct:.2f}%)")
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_data, tokenizer)
    val_dataset = prepare_dataset(val_data, tokenizer)
    
    # Batch size settings:
    # - 16GB RAM: batch_size=1, grad_accum=16
    # - 48GB RAM: batch_size=4, grad_accum=4 (faster training)
    batch_size = 1  # Adjust based on your RAM
    grad_accum = 16  # Effective batch = batch_size * grad_accum
    
    sft_config = SFTConfig(
        output_dir=str(OUTPUT_DIR / exp_name / "checkpoints"),
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=config["learning_rate"],
        warmup_ratio=0.03,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=DEVICE == "cuda",
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        dataloader_drop_last=True,
        max_length=512,
        dataset_text_field="text",
        packing=False,
    )
    
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
    )
    
    # Move to device if MPS
    if DEVICE == "mps":
        model = model.to("mps")
    
    # Train
    print(f"\nStarting training ({config['num_epochs']} epochs)...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time/60:.1f} minutes")
    
    # Save model
    output_dir = OUTPUT_DIR / exp_name
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_model(model, tokenizer, val_data)
    
    # Compile results
    results = {
        "experiment": exp_name,
        "name": exp_config["name"],
        "method": method,
        "config": config,
        "trainable_params": trainable_params,
        "trainable_pct": trainable_pct,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "metrics": metrics,
    }
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 Results:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.1f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  Time:      {training_time/60:.1f} min")
    
    # Cleanup
    del model
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results


def generate_comparison_report(all_results):
    """Generate a comparison report of all experiments."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Sort by F1 score
    sorted_results = sorted(all_results, key=lambda x: x["metrics"]["f1"], reverse=True)
    
    print("\n" + "="*80)
    print("📊 EXPERIMENT COMPARISON REPORT")
    print("="*80)
    
    # Table header
    print(f"\n{'Method':<30} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Time':>10} {'Params':>12}")
    print("-"*80)
    
    for r in sorted_results:
        m = r["metrics"]
        print(f"{r['name']:<30} {m['accuracy']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {r['training_time_minutes']:>8.1f}m {r['trainable_params']:>12,}")
    
    # Best results
    print("\n" + "="*80)
    print("🏆 BEST RESULTS")
    print("="*80)
    
    best = sorted_results[0]
    print(f"\nBest Method: {best['name']}")
    print(f"  Accuracy:  {best['metrics']['accuracy']:.4f}")
    print(f"  F1 Score:  {best['metrics']['f1']:.4f}")
    print(f"  Training:  {best['training_time_minutes']:.1f} minutes")
    print(f"  Params:    {best['trainable_params']:,} ({best['trainable_pct']:.2f}%)")
    
    # Save full comparison
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_ID,
        "device": DEVICE,
        "experiments": all_results,
        "best_experiment": best["experiment"],
    }
    
    comparison_path = RESULTS_DIR / "experiment_comparison.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n📁 Full comparison saved to: {comparison_path}")
    
    return comparison


def main():
    """Run all experiments and compare."""
    print("\n" + "="*80)
    print("🔬 LLAMA FINE-TUNING EXPERIMENT SUITE")
    print("="*80)
    print(f"\nModel: {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Experiments to run: {len(EXPERIMENTS)}")
    
    # Load data
    print("\nLoading data...")
    train_data = load_data(TRAIN_JSONL)
    val_data = load_data(VAL_JSONL)
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    
    # Run experiments
    all_results = []
    
    for exp_name, exp_config in EXPERIMENTS.items():
        try:
            results = run_experiment(exp_name, exp_config, train_data, val_data)
            all_results.append(results)
        except Exception as e:
            print(f"\n❌ Experiment {exp_name} failed: {e}")
            continue
    
    # Generate comparison
    if all_results:
        comparison = generate_comparison_report(all_results)
        
        print("\n" + "="*80)
        print("✅ ALL EXPERIMENTS COMPLETE!")
        print("="*80)
    else:
        print("\n❌ No experiments completed successfully.")


if __name__ == "__main__":
    main()
