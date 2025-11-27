"""
Prepare training data for stuttering variability classification.
Reads the PRISMA CSV, loads markdown files, cleans and chunks them,
then creates stratified train/val JSONL files.
"""

import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

from text_utils import clean_markdown, chunk_text


# Paths
BASE_DIR = Path(__file__).parent.parent
CSV_PATH = BASE_DIR / "PRISMA" / "output.csv"
MARKDOWN_DIR = BASE_DIR / "markdown"
OUTPUT_DIR = Path(__file__).parent


def load_csv(csv_path: Path) -> List[Dict]:
    """Load the PRISMA CSV file."""
    rows = []
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            print(f"Successfully loaded CSV with {encoding} encoding")
            return rows
        except UnicodeDecodeError:
            rows = []
            continue
    
    raise ValueError(f"Could not decode CSV with any encoding")


def map_label(outcome_value: str) -> int:
    """Map outcome to binary label. Accounted=1, else=0."""
    if outcome_value and outcome_value.strip().lower() == "accounted":
        return 1
    return 0


def read_markdown_file(markdown_dir: Path, filename: str) -> str:
    """Read a markdown file."""
    filepath = markdown_dir / filename
    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        return ""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def process_documents(csv_rows: List[Dict], markdown_dir: Path) -> List[Dict]:
    """Process all documents: read, clean, chunk."""
    entries = []
    
    # Find the outcome variability column
    outcome_col = None
    if csv_rows:
        for key in csv_rows[0].keys():
            if "Outcome Variability" in key and "Overall" in key:
                outcome_col = key
                break
    
    if not outcome_col:
        raise ValueError("Could not find Outcome Variability column")
    
    print(f"Using column: '{outcome_col}'")
    
    for row in csv_rows:
        source_file = row.get('source_file', '').strip()
        if not source_file:
            continue
        
        doc_id = Path(source_file).stem
        outcome_value = row.get(outcome_col, '')
        label = map_label(outcome_value)
        
        raw_text = read_markdown_file(markdown_dir, source_file)
        if not raw_text:
            continue
        
        cleaned_text = clean_markdown(raw_text)
        if not cleaned_text:
            continue
        
        chunks = chunk_text(cleaned_text, chunk_size=600, overlap=50)
        
        for i, chunk in enumerate(chunks):
            entry = {
                "id": f"{doc_id}_chunk_{i}",
                "doc_id": doc_id,
                "text": chunk,
                "label": label
            }
            entries.append(entry)
    
    return entries


def stratified_split(entries: List[Dict], test_size: float = 0.1, random_state: int = 42):
    """Stratified split at document level."""
    doc_entries = {}
    doc_labels = {}
    
    for entry in entries:
        doc_id = entry['doc_id']
        if doc_id not in doc_entries:
            doc_entries[doc_id] = []
            doc_labels[doc_id] = entry['label']
        doc_entries[doc_id].append(entry)
    
    doc_ids = list(doc_entries.keys())
    labels = [doc_labels[doc_id] for doc_id in doc_ids]
    
    train_doc_ids, val_doc_ids = train_test_split(
        doc_ids, test_size=test_size, stratify=labels, random_state=random_state
    )
    
    train_entries = []
    val_entries = []
    
    for doc_id in train_doc_ids:
        train_entries.extend(doc_entries[doc_id])
    for doc_id in val_doc_ids:
        val_entries.extend(doc_entries[doc_id])
    
    return train_entries, val_entries


def save_jsonl(entries: List[Dict], filepath: Path):
    """Save entries to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    print("=" * 60)
    print("Preparing training data for stuttering variability classifier")
    print("=" * 60)
    
    print(f"\nCSV path: {CSV_PATH}")
    print(f"Markdown directory: {MARKDOWN_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    print("\nLoading CSV...")
    csv_rows = load_csv(CSV_PATH)
    print(f"Loaded {len(csv_rows)} rows from CSV")
    
    print("\nProcessing documents...")
    entries = process_documents(csv_rows, MARKDOWN_DIR)
    print(f"Created {len(entries)} chunk entries")
    
    # Count labels
    label_counts = {0: 0, 1: 0}
    for entry in entries:
        label_counts[entry['label']] += 1
    print(f"Label distribution: 0={label_counts[0]}, 1={label_counts[1]}")
    
    # Stratified split
    print("\nPerforming stratified 90/10 split...")
    train_entries, val_entries = stratified_split(entries, test_size=0.1)
    
    print(f"Training chunks: {len(train_entries)}")
    print(f"Validation chunks: {len(val_entries)}")
    
    # Save
    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"
    
    print(f"\nSaving to {train_path}...")
    save_jsonl(train_entries, train_path)
    
    print(f"Saving to {val_path}...")
    save_jsonl(val_entries, val_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

