#!/usr/bin/env python3
"""
Convert all PDFs in a folder to Markdown into a new 'markdown' subfolder.

Best quality path: PyMuPDF (fitz) -> Markdown
Fallback path: pdfminer.six -> plain text saved as .md

Install (recommended):
    pip install pymupdf
Fallback:
    pip install pdfminer.six
"""

from pathlib import Path
import sys

# --- CONFIG: change if needed ---
BASE_DIR = Path("/Users/sivaguganjayachandran/Downloads/prisma")
OUT_DIR = BASE_DIR / "markdown"
# --------------------------------

# Try PyMuPDF first (better Markdown), else pdfminer.six (text)
try:
    import fitz  # PyMuPDF
    HAVE_PYMUPDF = True
except Exception:
    HAVE_PYMUPDF = False

try:
    from pdfminer.high_level import extract_text
    HAVE_PDFMINER = True
except Exception:
    HAVE_PDFMINER = False


def convert_with_pymupdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    parts = []
    for i, page in enumerate(doc):
        parts.append(page.get_text("markdown"))
        if i < len(doc) - 1:
            parts.append("\n\n---\n\n")  # page separator
    doc.close()
    return "".join(parts).strip() or "# (Empty)\n"


def convert_with_pdfminer(pdf_path: Path) -> str:
    txt = extract_text(str(pdf_path)) or ""
    # Light touch “markdown-ify”: keep it simple.
    md = f"# {pdf_path.stem}\n\n```\n{txt.strip()}\n```"
    return md


def main():
    if not BASE_DIR.exists():
        print(f"ERROR: Base folder not found: {BASE_DIR}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(exist_ok=True)

    pdfs = sorted(p for p in BASE_DIR.glob("*.pdf") if p.is_file())
    if not pdfs:
        print("No PDFs found. Nothing to do.")
        return

    if not HAVE_PYMUPDF and not HAVE_PDFMINER:
        print(
            "ERROR: Neither PyMuPDF nor pdfminer.six is installed.\n"
            "Install one of:\n"
            "  pip install pymupdf\n"
            "  pip install pdfminer.six",
            file=sys.stderr,
        )
        sys.exit(2)

    total = len(pdfs)
    ok, fail = 0, 0

    for idx, pdf in enumerate(pdfs, 1):
        out_path = OUT_DIR / (pdf.stem + ".md")
        try:
            if HAVE_PYMUPDF:
                md = convert_with_pymupdf(pdf)
            else:
                md = convert_with_pdfminer(pdf)

            out_path.write_text(md, encoding="utf-8")
            ok += 1
            print(f"[{idx}/{total}] ✔ {pdf.name} -> {out_path.name}")
        except Exception as e:
            fail += 1
            print(f"[{idx}/{total}] ✖ {pdf.name} (error: {e})", file=sys.stderr)

    print(f"\nDone. Success: {ok}, Failed: {fail}. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
