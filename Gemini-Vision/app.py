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
    """
    Determines the MIME type of the uploaded file and prepares it for the API.

    Args:
        file_path (str): The name of the uploaded file.
        uploaded_file (UploadedFile): The uploaded file object from Streamlit.

    Returns:
        list: A list containing a dictionary with MIME type and binary data.
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        mime_type = "image/jpeg"  # Fallback to a default type

    return [{
        "mime_type": mime_type,
        "data": uploaded_file.read()
    }]

def gemini_output(file_path, uploaded_file, system_prompt, user_prompt):
    """
    Sends the image and prompts to the Gemini-Vision model and retrieves the response.

    Args:
        file_path (str): The name of the uploaded file.
        uploaded_file (UploadedFile): The uploaded file object from Streamlit.
        system_prompt (str): The system-level prompt guiding the model.
        user_prompt (str): The user-level prompt requesting specific information.

    Returns:
        GenerateContentResponse: The response object from the model.
    """
    image_info = image_format(file_path, uploaded_file)
    input_prompt = [system_prompt, image_info[0], user_prompt]
    response = model.generate_content(input_prompt)
    return response

def create_excel_report(json_data):
    """
    Creates an Excel report from the extracted JSON data.

    Args:
        json_data (dict): The JSON data containing the measurements.

    Returns:
        bytes: The binary content of the Excel file.
    """
    measurements_df = pd.DataFrame(json_data['measurements'])
    title = json_data.get("title", "STRING CURRENT MEASUREMENT REPORT")
    date = json_data.get("date", "21/08/24")

    with BytesIO() as b:
        with pd.ExcelWriter(b, engine='openpyxl') as writer:
            # Write title and date without headers
            title_df = pd.DataFrame({0: [title, date]})
            title_df.to_excel(writer, sheet_name='Report', index=False, header=False, startrow=0)

            # Write measurements starting from row 4 (index 3)
            measurements_df.to_excel(writer, sheet_name='Report', index=False, startrow=3)

        b.seek(0)
        return b.read()

def main():
    """
    The main function that builds the Streamlit app interface and handles user interactions.
    """
    # Set the page configuration to wide layout
    st.set_page_config(page_title="Handwritten Data Extraction App", layout="wide")
    st.title("📝 Handwritten Data Extraction and Reporting App")
    st.markdown("""
        This app is designed to extract information from handwritten data from images using OCR or vision and
        create an Excel report with the extracted data.
    """)
    st.markdown("---")

    # Sidebar for image upload and instructions
    st.sidebar.header("📂 Upload Image")
    st.sidebar.info("""
        - **Supported Formats:** JPG, JPEG, PNG, WEBP
        - **Image Quality Tips:**
            - Ensure handwriting is clear and legible.
            - Avoid shadows and glare.
            - Use high-resolution images.
    """)
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Define the system and user prompts
        system_prompt = """You are a specialist in comprehending handwritten data from images.
                           Input images in the form of tables will be provided to you titled "STRING CURRENT MEASUREMENT REPORT",
                           and your task is to extract the handwritten digits using OCR or vision & 
                           respond to questions based on the content of the input image."""

        user_prompt = """
        Extract the information from the image and provide it in a JSON format with the following structure:

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
        }

        Ensure that all necessary fields are included and properly labeled.
        """

        with st.spinner("Processing image..."):
            try:
                # Reset the file pointer to read the data again
                uploaded_file.seek(0)
                # Send the image and prompts to the model
                response = gemini_output(uploaded_file.name, uploaded_file, system_prompt, user_prompt)

                # Extract the first candidate's content
                generated_content = response.candidates[0].content.parts[0].text

                # Clean and parse JSON
                cleaned_content = generated_content.strip("```json\n").strip("```")
                json_data = json.loads(cleaned_content)

                # Process and display key metrics
                measurements = json_data.get("measurements", [])
                if measurements:
                    # Extract Total Current values
                    total_currents = [entry.get("Total Current", 0) for entry in measurements]
                    highest_total_current = max(total_currents)
                    # Display the Date from the JSON
                    date = json_data.get("date", "Unknown Date")

                    # Show the Date and Highest Total Current
                    st.write(f"**📅 Date:** {date}  \n**🔍 Highest Total Current observed:** {highest_total_current}")

                    # Create and display DataFrame with Time, Strings 1-7, and Total Current
                    df = pd.DataFrame(measurements)
                    st.subheader("📊 STRING CURRENT MEASUREMENT REPORT")
                    st.dataframe(df[['Time', 'string-1', 'string-2', 'string-3', 'string-4', 'string-5', 'string-6', 'string-7', 'Total Current']], use_container_width=True)

                    # Provide download option for Excel report below the table
                    excel_data = create_excel_report(json_data)
                    st.download_button(
                        label="📥 Download Excel Report",
                        data=excel_data,
                        file_name='measurement_report.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                    )

                    # Display the complete JSON response last
                    st.subheader("📄 Complete JSON Response")
                    st.json(json_data)
                    
                    st.success("✅ Data extracted successfully!")
                else:
                    st.warning("⚠️ No measurements data found in the JSON response.")

            except json.JSONDecodeError as e:
                st.error(f"❌ Error decoding JSON: {e}")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

    else:
        st.info("ℹ️ Please upload an image file to get started.")

if __name__ == "__main__":
    main()
