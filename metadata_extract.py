import os

os.chdir("..")

from dotenv import load_dotenv
load_dotenv()

import csv
import ast
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Define the CSV column headers (index: name)
CSV_COLUMNS = {
    1: "Citation (APA)",
    2: "Participants (number, demographics, age, etc.)",
    3: "Study Type (e.g., RCT, observational, etc.)",
    4: "Self-ID (Yes / No / N/A / Not Reported)",
    5: "Inclusion stuttering behavioral criteria",
    6: "Standardized assessment used for inclusion (Y/N + list tools)",
    7: "Behavioral measure used for inclusion (Y/N + list tools)",
    8: "Behavioral measure used for outcome variable (Y/N + list tools)",
    9: "Study qualifies? (Yes on G or H criteria?)",
    10: "Behavioral measures for outcomes",
    11: "Extra behavioral measures",
    12: "Other measures (e.g., imaging, CAEPs, neuro data)",
    13: "Speech sample used for inclusion only? (Y/N)",
    14: "Speech sample used for grouping/categorization? (Y/N)",
    15: "Speech Sample A (type or description)",
    16: "Speech Sample B (type or description)",
    17: "Speech Sample C (type or description)",
    18: "Accounted for Time? (Y/N)",
    19: "Accounted for Task? (Y/N)",
    20: "Accounted for Setting? (Y/N)",
    21: "Potentially ignored variability",
    22: "Accounted for variability",
    23: "Ignored variability",
    24: "Notes 1",
    25: "Notes 2",
    26: "Questions",
    27: "Reviewer",
    28: "Date Reviewed",
    29: "Abstract"
}

# Escape all literal braces by doubling {{ }}
TEMPLATE_PROMPT = """
You are a highly precise research assistant tasked with extracting structured metadata from stuttering research articles.

Return only a Python dictionary with keys as integers (matching the field index below) and values as the extracted content.
Do not return any additional text, explanations, or commentary—only the dictionary. Ensure the output is a syntactically correct Python dictionary.

### Formatting rules:
- Every value must be a **single flat string**.
- If multiple items are needed, join them with "; " inside that string.
- Do **NOT** use lists, nested dictionaries, or JSON structures anywhere.
- Do **NOT** include quotation marks inside values unless they appear verbatim in the text.
- If data is missing or unclear, write "Not Reported".
- For Y/N questions, answer "Y" or "N" unless truly unknown (then "Not Reported").
- Keep each value concise (≤300 characters).
- Abstract (29) ≤ 3 sentences.
- Return ONLY the dictionary — nothing else.


Fields to extract:
 1. Citation (APA)
 2. Participants (number, demographics, age, etc.)
 3. Study Type (e.g., RCT, observational, etc.)
 4. Self-ID (Yes / No / N/A / Not Reported)
 5. Inclusion stuttering behavioral criteria
 6. Standardized assessment used for inclusion (Y/N + list tools)
 7. Behavioral measure used for inclusion (Y/N + list tools)
 8. Behavioral measure used for outcome variable (Y/N + list tools)
 9. Study qualifies? (Yes on G or H criteria?)
10. Behavioral measures for outcomes
11. Extra behavioral measures
12. Other measures (e.g., imaging, CAEPs, neuro data)
13. Speech sample used for inclusion only? (Y/N)
14. Speech sample used for grouping/categorization? (Y/N)
15. Speech Sample A (type or description)
16. Speech Sample B (type or description)
17. Speech Sample C (type or description)
18. Accounted for Time? (Y/N)
19. Accounted for Task? (Y/N)
20. Accounted for Setting? (Y/N)
21. Potentially ignored variability
22. Accounted for variability
23. Ignored variability
24. Notes 1
25. Notes 2
26. Questions
27. Reviewer - ChatGPT
28. Date Reviewed (use today's date in YYYY-MM-DD format)
29. Abstract (optional)

Article text starts below:
{text}
"""

def query_openai(text):
    prompt = TEMPLATE_PROMPT.format(text=text)
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API Error: {e}"

import re, json
from datetime import datetime

def _normalize_value(v):
    # list -> "a; b; c"
    if isinstance(v, list):
        return "; ".join(_normalize_value(x) for x in v if x is not None)
    # dict -> "k1: v1; k2: v2"
    if isinstance(v, dict):
        parts = []
        for k, val in v.items():
            parts.append(f"{k}: {_normalize_value(val)}")
        return "; ".join(parts)
    # everything else -> str
    return "" if v is None else str(v)

def _strip_code_fences(s: str) -> str:
    s = s.strip()
    # ```python ... ``` or ``` ... ```
    if s.startswith("```"):
        s = re.sub(r"^```(?:\w+)?\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()

def parse_dict_response(response: str):
    text = _strip_code_fences(response)

    # Try JSON first (model might return valid JSON)
    parsed_dict = None
    try:
        parsed_dict = json.loads(text)
    except Exception:
        # Fallback: Python literal
        try:
            parsed_dict = ast.literal_eval(text)
        except Exception as e:
            print(f"Error parsing response: {e}")
            # Initialize dict with "Not Reported"
            return {i: "Not Reported" for i in CSV_COLUMNS.keys()}

    if not isinstance(parsed_dict, dict):
        return {i: "Not Reported" for i in CSV_COLUMNS.keys()}

    # Initialize with "Not Reported"
    data_dict = {i: "Not Reported" for i in CSV_COLUMNS.keys()}

    for key, val in parsed_dict.items():
        try:
            key_int = int(key)
        except Exception:
            continue
        if key_int in CSV_COLUMNS:
            data_dict[key_int] = _normalize_value(val).strip()

    # Reviewer default (optional)
    if data_dict[27] in ("", "Not Reported"):
        data_dict[27] = "ChatGPT"

    # Date Reviewed default
    if data_dict[28] in ("", "Not Reported"):
        data_dict[28] = datetime.today().strftime('%Y-%m-%d')

    return data_dict




def save_dict_to_csv(data_dict, output_path="PRISMA/output.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_exists = os.path.isfile(output_path)

    # Write dict keys by index but headers by name in CSV
    with open(output_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[CSV_COLUMNS[i] for i in sorted(CSV_COLUMNS.keys())])
        if not file_exists:
            writer.writeheader()

        # Map dict with int keys to dict with column names
        row = {CSV_COLUMNS[i]: data_dict[i] for i in sorted(CSV_COLUMNS.keys())}
        writer.writerow(row)