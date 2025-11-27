"""
Run inference on markdown files using trained RoBERTa model.
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from text_utils import clean_markdown, chunk_text


# Default paths
DEFAULT_MODEL = Path(__file__).parent / "models" / "roberta_var"


def load_model(model_path: Path):
    """Load trained model and tokenizer."""
    print(f"Loading model from {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, device


def predict_document(model, tokenizer, device, text: str, threshold: float = 0.5):
    """Predict label for a document."""
    # Clean and chunk
    cleaned = clean_markdown(text)
    chunks = chunk_text(cleaned, chunk_size=600, overlap=50)
    
    if not chunks:
        return 0, 0.0
    
    # Get predictions for all chunks
    scores = []
    
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            score = probs[0, 1].item()  # Probability of class 1
            scores.append(score)
    
    # Average score across chunks
    avg_score = sum(scores) / len(scores)
    prediction = 1 if avg_score >= threshold else 0
    
    return prediction, avg_score


def main():
    parser = argparse.ArgumentParser(description="Run inference on markdown files")
    parser.add_argument("input", type=str, help="Input file or directory")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Model path")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--output", type=str, default="predictions.jsonl", help="Output file")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(Path(args.model))
    
    # Get input files
    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("*.md"))
    
    print("=" * 60)
    print("Stuttering Variability Classifier - Inference")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")
    print(f"Input files: {len(files)}")
    print("=" * 60)
    
    # Process files
    results = []
    print("\nProcessing files...")
    
    for file_path in files:
        print(f"Processing: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        pred, score = predict_document(model, tokenizer, device, text, args.threshold)
        
        label_str = "Accounted" if pred == 1 else "Not Accounted"
        print(f"  -> {label_str} (score: {score:.4f})")
        
        results.append({
            "file": file_path.name,
            "prediction": pred,
            "score": score,
            "label": label_str
        })
    
    # Save results
    with open(args.output, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    
    # Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Total documents: {len(results)}")
    print(f"  Accounted (1): {sum(1 for r in results if r['prediction'] == 1)}")
    print(f"  Not Accounted (0): {sum(1 for r in results if r['prediction'] == 0)}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
