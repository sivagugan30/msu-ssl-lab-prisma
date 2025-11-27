"""
Configuration for LLM fine-tuning with Llama 3.2.
"""

from pathlib import Path
import torch

# =============================================================================
# MODEL OPTIONS - Llama 3.2 1B
# =============================================================================

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# =============================================================================
# DEVICE CONFIGURATION (Auto-detect Mac MPS or CPU)
# =============================================================================
def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    else:
        return "cpu"

DEVICE = get_device()

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent.parent
ROBERTA_DIR = BASE_DIR / "roberta_classifier"
LLM_DIR = Path(__file__).parent

TRAIN_JSONL = ROBERTA_DIR / "train.jsonl"
VAL_JSONL = ROBERTA_DIR / "val.jsonl"

OUTPUT_DIR = LLM_DIR / "models"
RESULTS_DIR = LLM_DIR / "results"

# =============================================================================
# TRAINING HYPERPARAMETERS (optimized for 16GB RAM, no quantization)
# =============================================================================
TRAINING_CONFIG = {
    # LoRA settings
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    
    # Training settings
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "max_seq_length": 512,
    "warmup_ratio": 0.03,
    
    # Memory optimization
    "gradient_checkpointing": True,
    "fp16": False,
    "bf16": torch.cuda.is_available() or torch.backends.mps.is_available(),
}

# =============================================================================
# PROMPT TEMPLATE
# =============================================================================
SYSTEM_PROMPT = """You are a scientific paper analyzer specializing in stuttering research methodology.
Your task is to determine whether a research paper properly accounts for stuttering variability.

Stuttering variability means the paper measures or accounts for:
- Time variability: Different time points, sessions, or days
- Task variability: Different speaking tasks or conditions  
- Setting variability: Different environments or contexts

Respond with ONLY "accounted" or "not_accounted"."""

def format_prompt(text: str) -> str:
    """Format a text sample as a prompt for classification."""
    return f"""Analyze this excerpt from a stuttering research paper and determine if it accounts for stuttering variability.

Text:
{text[:2000]}

Does this paper account for stuttering variability? Answer ONLY with "accounted" or "not_accounted"."""


def format_training_example(text: str, label: int) -> str:
    """Format a training example with the expected output."""
    prompt = format_prompt(text)
    response = "accounted" if label == 1 else "not_accounted"
    return prompt, response
