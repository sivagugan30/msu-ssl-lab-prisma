"""
Metadata Extraction Pipeline for Stuttering Research Articles
Author: Siva
Purpose:
  - Query OpenAI to extract structured metadata fields (1–41)
  - Parse and normalize the model’s Python dictionary response
  - Save extracted fields into a clean CSV file
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
# 1. Environment Setup
# =======================
os.chdir("..")  # Move to project root if script is in a subfolder
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("❌ Missing OPENAI_API_KEY in environment variables")

client = OpenAI(api_key=openai_api_key)


# =======================
# 2. CSV Field Mapping
# =======================
CSV_COLUMNS = {
    1: "Citation (APA)",
    2: "Participants (number, demographics, age, etc.)",
    3: "Study Type (e.g., RCT, observational, etc.)",
    4: "Self-ID (Yes / No / N/A / Not Reported)",
    5: "Inclusion stuttering behavioral criteria",
    6: "Standardized assessment used for inclusion (Y/N + list tools)",
    7: "Behavioral measure used for inclusion (Y/N + list tools)",
    8: "Behavioral measure used for grouping/classification (Y/N + list tools)",
    9: "Behavioral measure used for outcome variable (Y/N + list tools)",
    10: "Study qualifies? (Yes / No / Not Reported)",
    11: "Additional stuttering measures (Y/N + examples)",
    12: "Speech sample details for inclusion (time/task/setting)",
    13: "Inclusion variability across time (Y/N/Not Reported)",
    14: "Number of time points for inclusion",
    15: "Inclusion variability across task (Y/N/Not Reported)",
    16: "Number of tasks for inclusion",
    17: "Inclusion variability across setting (Y/N/Not Reported)",
    18: "Number of settings for inclusion",
    19: "Inclusion accounted for variability (accounted/not accounted/undetermined)",
    20: "Speech samples for grouping/categorization (Y/N)",
    21: "Speech sample details for grouping (time/task/setting)",
    22: "Grouping variability across time (Y/N/Not Reported)",
    23: "Number of time points for grouping",
    24: "Grouping variability across task (Y/N/Not Reported)",
    25: "Number of tasks for grouping",
    26: "Grouping variability across setting (Y/N/Not Reported)",
    27: "Number of settings for grouping",
    28: "Grouping accounted for variability (accounted/not accounted/undetermined)",
    29: "Speech samples for outcome measures (Y/N)",
    30: "Speech sample details for outcome measures (time/task/setting)",
    31: "Outcome variability across time (Y/N/Not Reported)",
    32: "Number of time points for outcome measures",
    33: "Outcome variability across task (Y/N/Not Reported)",
    34: "Number of tasks for outcome measures",
    35: "Outcome variability across setting (Y/N/Not Reported)",
    36: "Number of settings for outcome measures",
    37: "Outcome accounted for variability (accounted/not accounted/undetermined)",
    38: "Additional notes",
    39: "Reviewer",
    40: "Date Reviewed",
    41: "Abstract (≤3 sentences)"
}


# =======================
# 3. LLM Extraction Prompt
# =======================
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
- Abstract (41) ≤ 3 sentences.
- Return ONLY the dictionary — nothing else.


Fields to extract:
Q1. Citation: In APA format, what is the full citation including title, authors, journal name, publication year, and DOI or URL?

Q2. Reviewer Info Your name:

Q3. Participants: How many participants were included, and what are their key demographics? (Example: 20 AWS and 20 AWNS)

Q4. Study Type: What is the type of study design? Select all that apply.
Options:

pre-post study

longitudinal

cross-sectional

experimental

case report

quasi-experimental

exploratory

randomized control trial

mixed methods

group comparison

qualitative

clinical trial

observational

survey

other

Q5. Self-Identification: Did participants self-identify as people who stutter?
Options:

Yes

No

Not applicable

Not reported

Q6. Inclusion Criteria for Stuttering: What specific stuttering-related criteria were used to include participants? (e.g., self-report, severity, %SS)
Criterion 1
Criterion 2
Criterion 3
Criterion 4
Criterion 5

Q7. Standardized Assessments: Were stuttering-specific standardized assessments for surface stuttering behaviors used for participant inclusion (e.g., SSI, TOCS)?
Options:

Yes

No

Q8. Please list the assessments.
Options:

SSI

TOCS

Other (please list)

Q9. Behavioral Measures for Inclusion: Were any stuttering behavioral measures used to determine inclusion? (e.g., %SS, types of disfluencies, etc.)
Options:

Yes

No

Not reported

Q10. Provide the specific behavioral measures (e.g., %SS, type and/or frequency of stuttering).
Measure 1
Measure 2
Measure 3
Measure 4
Measure 5

Q11. Behavioral Measures for grouping/classification: Were any stuttering behavioral measures used to group participants once they were included?
Options:

Yes

No

Not reported

Q12. Specify the grouping behavioral measures (e.g., %SS, 8-9-10 point severity scales, SSI)
Measure 1
Measure 2
Measure 3
Measure 4
Measure 5

Q13. Behavioral Measures as Outcome Variables: Were any stuttering behavioral measures used as outcome variables?
Options:

Yes

No

Not reported

Q14. Specify the outcome behavioral measures (e.g., %SS, 8-9-10 point severity scales, SSI)
Measure 1
Measure 2
Measure 3
Measure 4
Measure 5

Q15. Study Qualification (This study is qualified for review. Please continue the survey.)

Q16. Additional Stuttering Measures: Were any additional stuttering measures used to assess the affective or cognitive reactions of stuttering? (e.g., OASES, communicative participation, SESAS)
Options:

Yes

No

Not reported

Q17. Speech Sample Details for Inclusion: Please describe each speech sample used (e.g., time, task, and/or setting).
Speech Sample 1
Speech Sample 2
Speech Sample 3
Additional samples (if any)

Q18. Accounting for time in Inclusion Criteria: Did the study account for variability across time in participant inclusion? (must have used behavioral measures on at least 2 distinct days)
Options:

Yes

No

Not Reported or unclear (explain)

Q19. How many times? (inclusion criteria only)
Options:

1

2

3

4

5

More (Please Specify)

Q20. Accounting for task in Inclusion Criteria: Did the study account for variability across tasks in participant inclusion? (must have used behavioral measures in at least 2 distinct tasks, such as reading and conversational speech)
Options:

Yes

No

Not Reported or unclear (explain)

Q21. How many tasks? (inclusion criteria only)
Options:

1

2

3

4

5

More (Please Specify)

Q22. Accounting for Setting in Inclusion Criteria: Did the study account for variability across settings in participant inclusion? (must have used behavioral measures in at least 2 distinct settings, such as the clinic and at home)
Options:

Yes

No

Not Reported or unclear (explain)

Q23. How many settings? (inclusion criteria only)
Options:

1

2

3

4

5

More (Please Specify)

Q24. Accounting for Variability in Inclusion Criteria: Was variability accounted for in inclusion criteria in this study? (must have accounted for at least 2 of the 3: time, task, and/or setting)
Options:

Accounted for variability

Did not account for variability

Undetermined (Explain)

Q25. Speech Samples for Grouping/Categorizing: Were speech samples used for grouping participants once they were included in the study?
Options:

Yes

No

Q26. Speech Sample Details for Grouping/Categorization: Please describe each speech sample used (e.g., time, task, and/or setting).
Speech Sample 1
Speech Sample 2
Speech Sample 3
Additional samples (if any)

Q27. Accounting for time in Grouping/Categorization: Did the study account for variability across time in participant grouping? (must have used behavioral measures on at least 2 distinct days)
Options:

Yes

No

Not Reported or unclear (explain)

Q28. How many times? (participant grouping only)
Options:

1

2

3

4

5

More (Please Specify)

Q29. Accounting for task in Grouping/Categorization: Did the study account for variability across tasks in participant grouping? (must have used behavioral measures in at least 2 distinct tasks, such as reading and conversational speech)
Options:

Yes

No

Not Reported or unclear (explain)

Q30. How many tasks? (participant grouping only)
Options:

1

2

3

4

5

More (Please Specify)

Q31. Accounting for setting in Grouping/Categorization: Did the study account for variability across settings in participant grouping? (must have used behavioral measures in at least 2 distinct settings, such as the clinic and at home)
Options:

Yes

No

Not Reported or unclear (explain)

Q32. How many settings? (participant grouping only)
Options:

1

2

3

4

5

More (Please Specify)

Q33. Accounting for Variability in Grouping/Categorization: Was variability accounted for in participant grouping/categorization in this study? (must have accounted for at least 2 of the 3: time, task, and/or setting)
Options:

Accounted for variability

Did not account for variability

Undetermined (Explain)

Q34. Speech Samples for Outcome Measures: Were speech samples used for outcome measures in the study?
Options:

Yes

No

Q35. Speech Sample Details for Outcome Measures: Please describe each speech sample used (e.g., time, task, and/or setting).
Speech Sample 1
Speech Sample 2
Speech Sample 3
Additional samples (if any)

Q36. Accounting for time in Outcome Measures: Did the study account for variability across time in outcome measures? (must have used behavioral measures on at least 2 distinct days)
Options:

Yes

No

Not Reported or unclear (explain)

Q37. How many times? (outcome measures only)
Options:

1

2

3

4

5

More (Please Specify)

Q38. Accounting for task in Outcome Measures: Did the study account for variability across tasks in outcome measures? (must have used behavioral measures in at least 2 distinct tasks, such as reading and conversational speech)
Options:

Yes

No

Not Reported or unclear (explain)

Q39. How many tasks? (outcome measures only)
Options:

1

2

3

4

5

More (Please Specify)

Q40. Accounting for Setting in Outcome Measures: Did the study account for variability across settings in outcome measures? (must have used behavioral measures in at least 2 distinct settings, such as the clinic and at home)
Options:

Yes

No

Not Reported or unclear (explain)

Q41. How many settings? (outcome measures only)
Options:

1

2

3

4

5

More (Please Specify)

Q42. Accounting for Variability in Outcome Measures: Was variability accounted for in outcome measures in this study? (must have accounted for at least 2 of the 3: time, task, and/or setting)
Options:

Accounted for variability

Did not account for variability

Undetermined (Explain)

Q43. Additional Information and Notes: Is there any additional information or notes relevant to this review?

Article text starts below:
{text}
"""


# =======================
# 4. Query Function
# =======================
def query_openai(article_text: str) -> str:
    """Send article text to OpenAI and get structured metadata."""
    prompt = TEMPLATE_PROMPT.format(text=article_text)
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API Error: {e}"


# =======================
# 5. Parsing Helpers
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
    """Convert model response to normalized dictionary of metadata fields."""
    text = _strip_code_fences(response)
    try:
        parsed = json.loads(text)
    except Exception:
        try:
            parsed = ast.literal_eval(text)
        except Exception as e:
            print(f"⚠️ Error parsing response: {e}")
            parsed = {}

    data_dict = {i: "Not Reported" for i in CSV_COLUMNS.keys()}

    if isinstance(parsed, dict):
        for key, val in parsed.items():
            try:
                k = int(key)
                if k in CSV_COLUMNS:
                    data_dict[k] = _normalize_value(val)
            except Exception:
                continue

    # Default values
    data_dict[39] = data_dict.get(39, "ChatGPT") or "ChatGPT"
    data_dict[40] = datetime.today().strftime("%Y-%m-%d")

    return data_dict


# =======================
# 6. Save to CSV
# =======================
def save_dict_to_csv(data_dict, output_path="PRISMA/output_2.csv"):
    """Append the extracted metadata to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    file_exists = os.path.isfile(output_path)

    with open(output_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[CSV_COLUMNS[i] for i in sorted(CSV_COLUMNS.keys())])
        if not file_exists:
            writer.writeheader()

        row = {CSV_COLUMNS[i]: data_dict[i] for i in sorted(CSV_COLUMNS.keys())}
        writer.writerow(row)


# =======================
# 7. Example Usage
# =======================
if __name__ == "__main__":
    sample_text = """Example article abstract or full text here."""
    response = query_openai(sample_text)
    parsed = parse_dict_response(response)
    save_dict_to_csv(parsed)
    print("✅ Extraction complete. Data saved to PRISMA/output_2.csv")
