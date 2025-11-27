# LLM Fine-tuning for Stuttering Variability Classification

Fine-tune Llama 3.2-1B using LoRA for classifying whether research papers account for stuttering variability.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace
huggingface-cli login

# Run experiments
python experiment_runner.py
```

## For 48GB RAM (faster training)

Edit `experiment_runner.py`:
```python
batch_size = 4   # was 1
grad_accum = 4   # was 16
```
