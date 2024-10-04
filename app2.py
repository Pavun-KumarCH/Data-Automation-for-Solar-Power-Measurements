import os
import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from google.cloud import vision

# Set Google Cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'ServiceAccountToken.json'

# Initialize Google Vision client
client = vision.ImageAnnotatorClient()

def extract_text_from_image(image):
    """Extracts text from the uploaded image using Google Cloud Vision."""
    content = image.read()
    vision_image = vision.Image(content=content)
    response = client.text_detection(image=vision_image)
    
    if response.error.message:
        st.error(f"Error during text detection: {response.error.message}")
        return ""
    
    # Extract and return the detected text
    texts = response.text_annotations
    all_text = ""
    for text in texts:
        all_text += text.description + " "
    return all_text.strip()  # Remove any trailing spaces

def process_extracted_text(all_text):
    """Processes the extracted text into a DataFrame and extracts the date."""
    # Regex pattern for the main data
    pattern = r'(\d+:\d+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)'
    matches = re.findall(pattern, all_text)
    
    # Create DataFrame from extracted matches
    columns = ["Time", "String 1", "String 2", "String 3", "String 4", "String 5", "String 6", "String 7", "Total Current"]
    df = pd.DataFrame(matches, columns=columns)

    # Replace invalid values with NaN and convert to float
    def to_float(value):
        try:
            return float(value)
        except ValueError:
            return np.nan  # Replace with NaN if conversion fails

    # Apply the conversion to all relevant columns
    for col in columns[1:]:
        df[col] = df[col].apply(to_float)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Filter out rows with all NaN values or where 'Time' is NaN
    df = df.dropna(how='all')
    df = df[df['Time'].notna()]

    # Extract date using regex (assuming date format is dd/mm/yyyy or similar)
    date_pattern = r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b'  # Adjust this regex as per expected date formats
    date_matches = re.findall(date_pattern, all_text)
    
    # Get unique dates and return the DataFrame and the first date found
    unique_dates = list(set(date_matches))
    return df, unique_dates

# Title for the main content
st.title("String Current Measurement Report")
st.write("This application uses Google Cloud Vision to extract text from images and process it into a tabular format.")

# Sidebar for file upload
with st.sidebar:
    st.title("Upload & Settings")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# Main content area
if uploaded_file is not None:
    # Preview the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Extract text from the image
    extracted_text = extract_text_from_image(uploaded_file)

    if extracted_text:
        # Process the extracted text into a DataFrame and extract dates
        df, unique_dates = process_extracted_text(extracted_text)

        # Display the extracted dates at the top
        if unique_dates:
            st.subheader("Extracted Dates")
            for date in unique_dates:
                st.write(date)
        else:
            st.write("No dates found in the extracted text.")

        st.subheader("Extracted Text")
        st.write(extracted_text)

        # Display the cleaned DataFrame
        st.subheader("Data Table")
        st.dataframe(df)

        # Save DataFrame to Excel
        excel_file = io.BytesIO()
        df.to_excel(excel_file, index=False, sheet_name='Data', engine='openpyxl')
        excel_file.seek(0)  # Rewind the file pointer to the beginning

        st.download_button(
            label="Download Data Table as Excel",
            data=excel_file,
            file_name='data_table.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        # Save extracted text to TXT
        txt_file = io.StringIO()
        txt_file.write(extracted_text)
        txt_file.seek(0)  # Rewind the file pointer to the beginning

        st.download_button(
            label="Download Extracted Text as TXT",
            data=txt_file.getvalue(),
            file_name='extracted_text.txt',
            mime='text/plain'
        )

# Footer or additional information
st.write("This application uses Google Cloud Vision to extract text from images and process it into a tabular format.")
