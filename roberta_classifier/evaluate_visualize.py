"""
Generate evaluation visualizations and HTML report.
"""

import json
import csv
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


BASE_DIR = Path(__file__).parent
PREDICTIONS_PATH = BASE_DIR / "predictions.jsonl"
CSV_PATH = BASE_DIR.parent / "PRISMA" / "output.csv"
OUTPUT_HTML = BASE_DIR / "evaluation_report.html"


def load_ground_truth():
    """Load ground truth labels from CSV."""
    labels = {}
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(CSV_PATH, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    source_file = row.get('source_file', '').strip()
                    if not source_file:
                        continue
                    
                    # Find outcome column
                    outcome = None
                    for key in row.keys():
                        if "Outcome Variability" in key and "Overall" in key:
                            outcome = row[key]
                            break
                    
                    if outcome and outcome.strip().lower() == "accounted":
                        labels[source_file] = 1
                    else:
                        labels[source_file] = 0
            return labels
        except:
            continue
    
    return {}


def load_predictions():
    """Load predictions from JSONL."""
    preds = {}
    with open(PREDICTIONS_PATH, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            preds[data['file']] = {
                'prediction': data['prediction'],
                'score': data['score']
            }
    return preds


def generate_html_report(metrics, cm, matched):
    """Generate HTML report."""
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #333; }}
        .metric {{ display: inline-block; margin: 10px; padding: 20px; background: #e8f4f8; border-radius: 8px; text-align: center; }}
        .metric-value {{ font-size: 2em; font-weight: bold; color: #2196F3; }}
        .metric-label {{ color: #666; }}
        .cm {{ margin: 20px 0; }}
        .cm td {{ padding: 15px; text-align: center; border: 1px solid #ddd; }}
        .cm .tp {{ background: #c8e6c9; }}
        .cm .tn {{ background: #c8e6c9; }}
        .cm .fp {{ background: #ffcdd2; }}
        .cm .fn {{ background: #ffcdd2; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔬 Model Evaluation Report</h1>
        <p>Matched documents: {matched}</p>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{metrics['accuracy']:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['precision']:.1%}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['recall']:.1%}</div>
                <div class="metric-label">Recall</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['f1']:.1%}</div>
                <div class="metric-label">F1 Score</div>
            </div>
        </div>
        
        <h2>Confusion Matrix</h2>
        <table class="cm">
            <tr><td></td><td><b>Pred: Not Acc</b></td><td><b>Pred: Accounted</b></td></tr>
            <tr><td><b>True: Not Acc</b></td><td class="tn">TN: {cm[0][0]}</td><td class="fp">FP: {cm[0][1]}</td></tr>
            <tr><td><b>True: Accounted</b></td><td class="fn">FN: {cm[1][0]}</td><td class="tp">TP: {cm[1][1]}</td></tr>
        </table>
    </div>
</body>
</html>"""
    return html


def main():
    print("=" * 60)
    print("Generating Model Evaluation Visualizations")
    print("=" * 60)
    
    # Load data
    print("\nLoading ground truth labels...")
    ground_truth = load_ground_truth()
    print(f"Loaded {len(ground_truth)} ground truth labels")
    
    print("\nLoading predictions...")
    predictions = load_predictions()
    print(f"Loaded {len(predictions)} predictions")
    
    # Match
    y_true = []
    y_pred = []
    
    for filename, gt_label in ground_truth.items():
        if filename in predictions:
            y_true.append(gt_label)
            y_pred.append(predictions[filename]['prediction'])
    
    print(f"\nMatched documents: {len(y_true)}")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n--- Results ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {cm[1][1]}  FN: {cm[1][0]}")
    print(f"  FP: {cm[0][1]}  TN: {cm[0][0]}")
    
    # Generate HTML
    print("\nGenerating HTML report...")
    html = generate_html_report(metrics, cm.tolist(), len(y_true))
    
    with open(OUTPUT_HTML, 'w') as f:
        f.write(html)
    
    print(f"Report saved to: {OUTPUT_HTML}")
    print("\n" + "=" * 60)
    print("Done!")
    print(f"Open {OUTPUT_HTML} in your browser to view the report.")
    print("=" * 60)


if __name__ == "__main__":
    main()
