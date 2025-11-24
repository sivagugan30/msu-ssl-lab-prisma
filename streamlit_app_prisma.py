from metadata_extract_2 import query_openai, parse_dict_response, save_dict_to_csv, CSV_COLUMNS
import streamlit as st
import time
from dotenv import load_dotenv
import os
import os

# Save outputs inside the repo (e.g., PRISMA/output.csv)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "PRISMA", "output.csv")


# Load the .env file
load_dotenv()

# Access the value of key1
key1 = os.getenv("key1")

# APP_PASSWORD = os.getenv("APP_PASSWORD")  # set this in your .env file
#
# # --- PASSWORD GATE ---
# def check_password():
#     st.title("🔒 Secure Login")
#
#     # Ask for password
#     password = st.text_input("Enter password:", type="password")
#
#     if password == APP_PASSWORD:
#         st.session_state["authenticated"] = True
#         st.success("Access granted ✅")
#         return True
#     elif password:
#         st.error("Incorrect password ❌")
#         return False
#     else:
#         return False
#
# # --- MAIN APP ---
# if "authenticated" not in st.session_state or not st.session_state["authenticated"]:
#     if not check_password():
#         st.stop()

def extract_text():
    st.title("Research Article Metadata Extraction")

    article_text = st.text_area(
        "Paste the full research article text here:",
        height=300,
        placeholder="Enter article content..."
    )

    if st.button("Extract Metadata"):
        if not article_text.strip():
            st.warning("Please provide the article text.")
            return

        with st.spinner("Extracting metadata..."):
            response = query_openai(article_text)
            if response.startswith("OpenAI API Error"):
                st.error(response)
                return

            metadata_dict = parse_dict_response(response)
            save_dict_to_csv(metadata_dict, output_path=CSV_PATH)

        st.balloons()
        abs_path = os.path.abspath(CSV_PATH)
        st.success(f"Metadata saved to '{abs_path}'.")
        st.snow()

        import pandas as pd

        named_dict = {CSV_COLUMNS[i]: v for i, v in metadata_dict.items()}
        df = pd.DataFrame(list(named_dict.items()), columns=["Field", "Value"])
        st.dataframe(df, use_container_width=True)

        with st.expander("Show extracted metadata (JSON view)", expanded=False):
            st.json(metadata_dict)
            st.write(response)


if __name__ == "__main__":
    extract_text()




