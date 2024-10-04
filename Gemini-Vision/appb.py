import os
import mimetypes
import json
from pathlib import Path
from io import BytesIO

import google.generativeai as genai
from dotenv import load_dotenv
import streamlit as st
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI with the API key
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Model Configuration
MODEL_CONFIG = {
    "temperature": 0.3,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
}

# Safety Settings of Model
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# Initialize the Gemini-Vision Model
model = genai.GenerativeModel(
    model_name='gemini-1.5-flash-002',
    generation_config=MODEL_CONFIG,
    safety_settings=SAFETY_SETTINGS
)

def image_format(file_path, uploaded_file):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "image/jpeg"
    return [{
        "mime_type": mime_type,
        "data": uploaded_file.read()
    }]

def gemini_output(file_path, uploaded_file, system_prompt, user_prompt):
    image_info = image_format(file_path, uploaded_file)
    input_prompt = [system_prompt, image_info[0], user_prompt]
    response = model.generate_content(input_prompt)
    return response

def create_excel_report(json_data):
    measurements_df = pd.DataFrame(json_data['measurements'])
    title = json_data.get("title", "STRING CURRENT MEASUREMENT REPORT")
    date = json_data.get("date", "21/08/24")

    with BytesIO() as b:
        with pd.ExcelWriter(b, engine='openpyxl') as writer:
            title_df = pd.DataFrame({0: [title, date]})
            title_df.to_excel(writer, sheet_name='Report', index=False, header=False, startrow=0)
            measurements_df.to_excel(writer, sheet_name='Report', index=False, startrow=3)
        b.seek(0)
        return b.read()

def main():
    st.set_page_config(page_title="Handwritten Data Extraction App", layout="wide")
    st.title("📝 Handwritten Data Extraction and Reporting App")
    st.markdown("---")

    st.sidebar.header("📂 Upload Image")
    st.sidebar.info("""Supported Formats: JPG, JPEG, PNG, WEBP""")
    uploaded_files = st.sidebar.file_uploader("Choose images...", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

    if uploaded_files:
        all_measurements = []
        highest_total_currents = []

        for uploaded_file in uploaded_files:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            system_prompt = """You are a specialist in comprehending handwritten data from images.
                               Input images in the form of tables will be provided to you titled "STRING CURRENT MEASUREMENT REPORT",
                               and your task is to extract the handwritten digits using OCR or vision &
                               respond to questions based on the content of the input image."""
            user_prompt = """Extract the information from the image and provide it in a JSON format with the following structure:
            {
                "title": "STRING CURRENT MEASUREMENT REPORT",
                "date": "DD/MM/YY",
                "measurements": [
                    {
                        "Time": <value>,
                        "string-1": <value>,
                        "string-2": <value>,
                        "string-3": <value>,
                        "string-4": <value>,
                        "string-5": <value>,
                        "string-6": <value>,
                        "string-7": <value>,
                        "Total Current": <value>
                    },
                    ...
                ]
            }"""

            with st.spinner("Processing image..."):
                try:
                    uploaded_file.seek(0)
                    response = gemini_output(uploaded_file.name, uploaded_file, system_prompt, user_prompt)
                    generated_content = response.candidates[0].content.parts[0].text
                    cleaned_content = generated_content.strip("```json\n").strip("```")
                    json_data = json.loads(cleaned_content)

                    measurements = json_data.get("measurements", [])
                    date = json_data.get("date", "Unknown Date")
                    if measurements:
                        total_currents = [entry.get("Total Current", 0) for entry in measurements]
                        highest_total_current = max(total_currents)
                        highest_total_currents.append((date, highest_total_current))
                        all_measurements.extend(measurements)
                    else:
                        st.warning("⚠️ No measurements data found in the JSON response.")

                except json.JSONDecodeError as e:
                    st.error(f"❌ Error decoding JSON: {e}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")

        # After processing all files
        if all_measurements:
            df_all = pd.DataFrame(all_measurements)

            # Print the DataFrame for inspection
            # st.write("Extracted DataFrame:", df_all)

            expected_columns = ['Time', 'string-1', 'string-2', 'string-3', 'string-4', 'string-5', 'string-6', 'string-7', 'Total Current']
            missing_columns = [col for col in expected_columns if col not in df_all.columns]

            if missing_columns:
                st.warning(f"⚠️ The following expected columns are missing: {missing_columns}")
            else:
                # Display the combined DataFrame
                st.subheader("📊 STRING CURRENT MEASUREMENT REPORT")
                st.dataframe(df_all[expected_columns], use_container_width=True)

                # Display the highest total currents observed
                for date, total in highest_total_currents:
                    st.write(f"📅 Date: {date}  \n🔍 Highest Total Current observed: {total}")

                # Create and display combined Excel report
                combined_json_data = {"measurements": all_measurements}
                combined_excel_data = create_excel_report(combined_json_data)
                st.download_button(
                    label="📥 Download Combined Excel Report",
                    data=combined_excel_data,
                    file_name='combined_measurement_report.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )
        else:
            st.warning("⚠️ No valid measurements found across the uploaded images.")

    else:
        st.info("ℹ️ Please upload at least one image file to get started.")

if __name__ == "__main__":
    main()
