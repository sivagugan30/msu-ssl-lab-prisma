# LLM Fine-tuning Setup Instructions

## 📋 Project Overview

This project fine-tunes **Llama 3.2-1B-Instruct** using LoRA for classifying whether research papers account for stuttering variability. It includes:

1. **RoBERTa Classifier** - Baseline encoder model (already trained)
2. **LLM Classifier** - Llama 3.2-1B with multiple LoRA fine-tuning experiments

## 🚀 Quick Start on M4 Pro Mac Mini (48GB RAM)

### Step 1: Clone Repository
```bash
git clone https://github.com/sivagugan30/msu-ssl-lab-prisma.git
cd msu-ssl-lab-prisma
```

### Step 2: Install Dependencies
```bash
pip install -r llm_classifier/requirements.txt
```

### Step 3: Authenticate HuggingFace
```bash
huggingface-cli login
# Enter your HuggingFace token when prompted
```

**Important:** Make sure you have access to `meta-llama/Llama-3.2-1B-Instruct`:
- Go to: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- Click "Agree and access repository" if you haven't already

### Step 4: Prepare Training Data
```bash
cd roberta_classifier
python prepare_data.py
```

This creates:
- `train.jsonl` - 1156 training samples
- `val.jsonl` - 97 validation samples

### Step 5: Optimize for 48GB RAM (Faster Training)

Edit `llm_classifier/experiment_runner.py` around line 244:

```python
# Change from:
batch_size = 1
grad_accum = 16

# To:
batch_size = 4   # 4x faster on 48GB RAM
grad_accum = 4   # Keep effective batch = 16
```

### Step 6: Run LLM Experiments
```bash
cd ../llm_classifier
python experiment_runner.py
```

This will run **4 experiments**:
1. **LoRA (r=16)** - Baseline, ~12 hours
2. **LoRA (r=64)** - High rank, more capacity, ~12 hours
3. **LoRA (High LR)** - Faster training, ~8 hours
4. **IA3** - Lightweight alternative, ~10 hours

**Total time:** ~42 hours with batch_size=1, or **~10 hours** with batch_size=4 on 48GB RAM

## 📁 Project Structure

```
msu-ssl-lab-prisma/
├── llm_classifier/
│   ├── config.py              # Model config (Llama 3.2-1B)
│   ├── experiment_runner.py   # Main script - runs 4 experiments
│   ├── requirements.txt        # Python dependencies
│   └── README.md              # Documentation
│
├── roberta_classifier/
│   ├── prepare_data.py        # Creates train/val JSONL files
│   ├── train_roberta.py        # Fine-tune RoBERTa (already done)
│   ├── infer.py                # Run inference
│   ├── evaluate_visualize.py   # Generate evaluation report
│   └── text_utils.py          # Text cleaning utilities
│
└── .gitignore                  # Excludes large model files
```

## 🔬 Experiments Explained

### Experiment 1: LoRA (r=16)
- **Rank:** 16 (baseline)
- **Alpha:** 32
- **Epochs:** 3
- **LR:** 2e-4
- **Trainable params:** ~11M (0.9% of model)

### Experiment 2: LoRA (r=64)
- **Rank:** 64 (higher capacity)
- **Alpha:** 128
- **Epochs:** 3
- **LR:** 2e-4
- **Trainable params:** ~45M (3.6% of model)

### Experiment 3: LoRA (High LR)
- **Rank:** 16
- **Epochs:** 2 (faster)
- **LR:** 5e-4 (higher learning rate)
- **Dropout:** 0.1 (more regularization)

### Experiment 4: IA3
- **Method:** IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)
- **Epochs:** 3
- **LR:** 3e-4
- **Trainable params:** ~1M (very lightweight)

## 📊 Expected Results

After experiments complete, you'll get:

1. **Individual results** in `llm_classifier/models/{experiment_name}/results.json`
2. **Comparison report** in `llm_classifier/results/experiment_comparison.json`
3. **Console output** showing:
   - Training time per experiment
   - Accuracy, Precision, Recall, F1 scores
   - Trainable parameter counts
   - Best performing method

## 🎯 Evaluation Method

The model uses **logit-based classification**:
- Compares probability of generating "accounted" vs "not" as next token
- More reliable than text generation for small LLMs
- Better than parsing free-form responses

## ⚙️ Configuration

### Model
- **Base:** `meta-llama/Llama-3.2-1B-Instruct`
- **Device:** Auto-detects MPS (Apple Silicon) or CUDA

### Training (16GB RAM default)
- Batch size: 1
- Gradient accumulation: 16
- Effective batch: 16
- Max sequence length: 512

### Training (48GB RAM optimized)
- Batch size: 4
- Gradient accumulation: 4
- Effective batch: 16
- **4x faster training**

## 📝 Notes

1. **Model files are NOT in git** - They're excluded by `.gitignore` to keep repo small
2. **Training data** comes from `roberta_classifier/train.jsonl` and `val.jsonl`
3. **Results** are saved in `llm_classifier/models/` and `llm_classifier/results/`
4. **Each experiment** saves its own model in `models/{experiment_name}/final/`

## 🐛 Troubleshooting

### "Gated repo" error
- Make sure you're logged in: `huggingface-cli login`
- Verify access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

### Out of memory
- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps` to compensate
- Reduce `max_seq_length` to 256

### Slow training
- On 48GB RAM, use `batch_size=4` for 4x speedup
- Consider running fewer experiments initially

## 📧 Contact

If you encounter issues, check:
1. HuggingFace authentication
2. Model access permissions
3. Training data exists (`train.jsonl`, `val.jsonl`)
4. Dependencies installed correctly

---

**Good luck with the experiments!** 🚀

