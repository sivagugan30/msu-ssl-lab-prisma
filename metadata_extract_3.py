"""
Minimal Variability Extraction (Outcome + Overall Flags)
Author: Siva

CSV columns (in this order):
1) Citation (APA)
2) Reviewer
3) Date Reviewed
4) Outcome variability — Time (Y/N/Not Reported)
5) Outcome variability — Task (Y/N/Not Reported)
6) Outcome variability — Setting (Y/N/Not Reported)
7) Outcome variability — Overall (Accounted/Not Accounted/Undetermined)
8) Inclusion variability — Overall (Accounted/Not Accounted/Undetermined)
9) Grouping variability — Overall (Accounted/Not Accounted/Undetermined)
"""

import os
import csv
import ast
import re
import json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# =======================
# 1) Environment
# =======================
# os.chdir("..")  # optional: move to project root
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# =======================
# 2) Columns (Citation/Reviewer/Date FIRST)
# =======================
CSV_COLUMNS = {
    7: "Citation (APA)",
    8: "Reviewer",
    9: "Date Reviewed",
    1: "Outcome variability — Time (Y/N/Not Reported)",
    2: "Outcome variability — Task (Y/N/Not Reported)",
    3: "Outcome variability — Setting (Y/N/Not Reported)",
    4: "Outcome variability — Overall (Accounted/Not Accounted/Undetermined)",
    5: "Inclusion variability — Overall (Accounted/Not Accounted/Undetermined)",
    6: "Grouping variability — Overall (Accounted/Not Accounted/Undetermined)",
}

# Preserve the explicit order above
FIELD_ORDER = [7, 8, 9, 1, 2, 3, 4, 5, 6]

# =======================
# 3) Detailed, contextual prompt
# =======================
TEMPLATE_PROMPT = r"""
You are a research extraction specialist. Extract ONLY the fields below from a stuttering research article.

CRITICAL CONTEXT (READ CAREFULLY)
- "Stuttering variability" refers to CHANGES in stuttering behavior/severity **as an outcome** across:
  • TIME (≥2 distinct sessions/days/visits; longitudinal, pre-post on different days)
  • TASK (≥2 distinct speaking tasks, e.g., reading vs conversation, monologue, picture description)
  • SETTING (≥2 distinct environments, e.g., clinic/lab vs home/school/telepractice)
- Outcome variability must be based on speech-related outcome measures (e.g., %SS, SSI, disfluency counts, severity scales).
- Do NOT count “inclusion” or “grouping” procedures as “outcomes.” Outcomes mean how results were measured/analyzed.
- Synonyms/examples:
  TIME → session(s), visit(s), day(s), follow-up(s), repeated measures across days.
  TASK → reading, conversation, monologue, spontaneous speech, narrative, picture description.
  SETTING → clinic/lab, home/naturalistic, school/classroom, telepractice vs in-person.
- Edge cases:
  • Single-session pre–post within the same day = NOT time variability.
  • Longitudinal single task/setting = TIME=Y, TASK=N, SETTING=N.
  • Two tasks on same day = TASK=Y.
  • Only self-report without speech outcomes = Not Reported for outcomes.
- OVERALL rules (apply separately to outcome vs inclusion vs grouping):
  • "Accounted"  = at least TWO of {time, task, setting} clearly present for that stage.
  • "Not Accounted" = zero or only one dimension present.
  • "Undetermined" = ambiguous/unclear/conflicting.

RETURN FORMAT (STRICT)
- Return ONLY a valid Python dict (no extra text).
- Keys are integers 1–9 (defined below). Values are SINGLE STRINGS (≤300 chars).
- If unknown, write exactly "Not Reported".
- For 1–3 answer EXACTLY: "Y", "N", or "Not Reported".
- For 4–6 answer EXACTLY: "Accounted", "Not Accounted", or "Undetermined".
- 7 (Citation): APA if available in text; otherwise "Not Reported".
- 8 (Reviewer): If article text contains a reviewer/authoring reviewer name, include it; otherwise "Not Reported".
- 9 (Date Reviewed): If publication year is stated, return just the 4-digit year; otherwise today's date YYYY-MM-DD.

FIELDS TO EXTRACT
1: Outcome variability across TIME  (≥2 distinct days/sessions/visits?) -> "Y"/"N"/"Not Reported"
2: Outcome variability across TASK  (≥2 distinct speaking tasks?)       -> "Y"/"N"/"Not Reported"
3: Outcome variability across SETTING (≥2 distinct environments?)       -> "Y"/"N"/"Not Reported"
4: Outcome variability OVERALL      (time/task/setting ≥2?)             -> "Accounted"/"Not Accounted"/"Undetermined"
5: Inclusion variability OVERALL    (during participant inclusion)      -> "Accounted"/"Not Accounted"/"Undetermined"
6: Grouping variability OVERALL     (during participant grouping)       -> "Accounted"/"Not Accounted"/"Undetermined"
7: Citation (APA)
8: Reviewer
9: Date Reviewed

EXAMPLES (for your internal guidance; do NOT echo back)
- If outcomes compare reading vs conversation within a single lab visit → 1=N, 2=Y, 3=N, 4=Not Accounted.
- If outcomes collect %SS across 3 weekly sessions in clinic only → 1=Y, 2=N, 3=N, 4=Not Accounted.
- If outcomes compare reading & conversation across lab and home on two days → 1=Y, 2=Y, 3=Y, 4=Accounted.

Article text starts below:
{text}
"""

# =======================
# 4) OpenAI call
# =======================
def query_openai(article_text: str) -> str:
    prompt = TEMPLATE_PROMPT.format(text=article_text)
    resp = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()

# =======================
# 5) Parsing helpers
# =======================
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:\w+)?\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    return s.strip()

def _normalize_value(v):
    if isinstance(v, list):
        return "; ".join(_normalize_value(x) for x in v if x)
    if isinstance(v, dict):
        return "; ".join(f"{k}: {_normalize_value(vv)}" for k, vv in v.items())
    return "" if v is None else str(v).strip()

def parse_dict_response(response: str):
    text = _strip_code_fences(response)
    parsed = {}
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception as e:
            print(f"⚠️ Parse error: {e}")

    # defaults
    data = {k: "Not Reported" for k in CSV_COLUMNS.keys()}

    if isinstance(parsed, dict):
        for key, val in parsed.items():
            try:
                k = int(key)
            except:
                continue
            if k in data:
                data[k] = _normalize_value(val)

    # Clamp vocab for 1–3 (Y/N/Not Reported)
    def clamp_yesno(v):
        v2 = (v or "").strip().lower()
        if v2 in {"y", "yes"}: return "Y"
        if v2 in {"n", "no"}:  return "N"
        if v2 == "not reported": return "Not Reported"
        return "Not Reported"

    # Clamp vocab for 4–6 (Accounted/Not Accounted/Undetermined)
    def clamp_overall(v):
        v2 = (v or "").strip().lower()
        if "not accounted" in v2 or v2 == "no":
            return "Not Accounted"
        if "accounted" in v2 and "not" not in v2:
            return "Accounted"
        if "undetermined" in v2:
            return "Undetermined"
        return "Undetermined" if v2 else "Undetermined"

    data[1] = clamp_yesno(data[1])
    data[2] = clamp_yesno(data[2])
    data[3] = clamp_yesno(data[3])
    data[4] = clamp_overall(data[4])
    data[5] = clamp_overall(data[5])
    data[6] = clamp_overall(data[6])

    # Reviewer fallback
    if not data[8] or data[8] == "Not Reported":
        data[8] = "ChatGPT"

    # Date Reviewed: if 9 looks like a 4-digit year, keep; else use today
    y = (data[9] or "").strip()
    if not (len(y) == 4 and y.isdigit()):
        data[9] = datetime.today().strftime("%Y-%m-%d")

    return data

# =======================
# 6) Save CSV
# =======================
def save_dict_to_csv(data_dict, output_path="PRISMA/variability_flags.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_exists = os.path.isfile(output_path)
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[CSV_COLUMNS[i] for i in FIELD_ORDER]
        )
        if not file_exists:
            writer.writeheader()
        row = {CSV_COLUMNS[i]: data_dict[i] for i in FIELD_ORDER}
        writer.writerow(row)

# =======================
# 7) Example usage
# =======================
if __name__ == "__main__":
    sample_text = """Paste abstract/full text here."""
    raw = query_openai(sample_text)
    parsed = parse_dict_response(raw)
    save_dict_to_csv(parsed)
    print("✅ Saved to PRISMA/variability_flags.csv")
