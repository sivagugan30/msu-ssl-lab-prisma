#!/usr/bin/env python3
import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ---- Your existing helpers ----
from metadata_extract_2 import (
    query_openai,
    parse_dict_response,
    save_dict_to_csv,  # kept for compatibility; not required for the batch save, but we won't remove it
    CSV_COLUMNS,
)

# --------------------------------
# Paths & env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MD_DIR = "/Users/sivaguganjayachandran/PycharmProjects/msu-ssl-lab/PRISMA/markdown-copy"
CSV_PATH = os.path.join(BASE_DIR, "PRISMA", "output.csv")
XLSX_PATH = os.path.join(BASE_DIR, "PRISMA", "output.xlsx")

load_dotenv()
_ = os.getenv("key1")  # if your query_openai uses env, keep this load

# --------------------------------
def read_markdown_files(md_dir: Path):
    """Yield (file_path, text) for all .md files (non-recursive)."""
    for p in sorted(md_dir.glob("*.md")):
        try:
            yield p, p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            st.error(f"Failed to read {p.name}: {e}")

def normalize_row(md: dict, filename: str):
    """
    Ensure row has all CSV_COLUMNS; add source file.
    Unknown keys are ignored; missing keys become ''.
    """
    row = {"source_file": filename}
    for col in CSV_COLUMNS:
        row[col] = md.get(col, "")
    return row

def batch_extract(md_dir: Path, delay_s: float = 0.0):
    """
    Process all .md files in md_dir using query_openai + parse_dict_response.
    Returns a DataFrame with columns: source_file + CSV_COLUMNS.
    """
    rows = []
    files = list(read_markdown_files(md_dir))
    if not files:
        return pd.DataFrame(columns=["source_file"] + list(CSV_COLUMNS))

    progress = st.progress(0)
    status = st.empty()

    for i, (path, text) in enumerate(files, start=1):
        status.write(f"Processing {path.name} ({i}/{len(files)})…")
        try:
            resp = query_openai(text)
            if isinstance(resp, str) and resp.startswith("OpenAI API Error"):
                st.warning(f"{path.name}: {resp}")
                continue

            md = parse_dict_response(resp)
            row = normalize_row(md, path.name)
            rows.append(row)

        except Exception as e:
            st.error(f"{path.name}: {e}")

        progress.progress(i / len(files))
        if delay_s > 0:
            time.sleep(delay_s)

    status.write("Done.")
    cols = ["source_file"] + list(CSV_COLUMNS)
    return pd.DataFrame(rows, columns=cols)

def save_outputs(df: pd.DataFrame, fmt: str):
    os.makedirs(os.path.join(BASE_DIR, "PRISMA"), exist_ok=True)
    if fmt == "CSV":
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        return CSV_PATH
    else:
        df.to_excel(XLSX_PATH, index=False, engine="openpyxl")
        return XLSX_PATH

# --------------------------------
def main():
    st.title("Batch Metadata Extraction from Markdown Folder")

    md_dir_input = st.text_input(
        "Markdown folder path:",
        value=DEFAULT_MD_DIR,
        help="Provide the folder that contains .md files (non-recursive).",
    )

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        out_fmt = st.selectbox("Output format", ["CSV", "Excel"], index=0)
    with col2:
        delay_s = st.number_input("Delay between files (s)", min_value=0.0, max_value=10.0, value=0.0, step=0.5)
    with col3:
        st.caption("Tip: add a small delay if you hit rate limits.")

    if st.button("Process Folder"):
        md_dir = Path(md_dir_input).expanduser()
        if not md_dir.exists() or not md_dir.is_dir():
            st.error(f"Folder not found: {md_dir}")
            st.stop()

        with st.spinner("Extracting metadata from all Markdown files…"):
            df = batch_extract(md_dir, delay_s=delay_s)

        if df.empty:
            st.warning("No rows extracted. Check the folder or parsing.")
            st.stop()

        # Show preview
        st.success(f"Extracted {len(df)} records.")
        st.dataframe(df, use_container_width=True)

        # Save combined output
        out_path = save_outputs(df, out_fmt)
        st.success(f"Saved {out_fmt} to: {os.path.abspath(out_path)}")

        # Optional: also keep a CSV copy with your existing helper
        try:
            # Append each row to the legacy CSV if you want; else skip.
            # Here we just write the combined DF once via pandas already.
            pass
        except Exception as e:
            st.warning(f"Legacy save_dict_to_csv step skipped: {e}")

if __name__ == "__main__":
    main()
