# Data-Automation-for-Solar-Power-Measurements

### Solar Panel Performance Automation Project

## Overview

This project addresses the challenges of manual data entry for solar panel performance monitoring, which can lead to errors, delays, and inefficiencies. By implementing Optical Character Recognition (OCR) and Object Detection technologies, we aim to automate the data preparation process, digitizing handwritten power measurements to enhance real-time analysis and operational management.

## Business Problem

Manual data entry for solar panel performance is prone to errors, which affects real-time analysis and operational management. This project seeks to eliminate those inefficiencies and provide a reliable solution for monitoring solar panel performance.

## Business Objectives

- **Automate the data preparation process** using OCR and Object Detection.
- **Reduce operational delays and errors** in data entry.
- **Improve operational efficiency by 80%**.
- **Achieve 95% accuracy in detecting faulty solar modules**.

## Scope

This project will primarily focus on the Operations Department, which monitors solar panel performance.

## Approach

The project follows the **CRISP-ML(Q)** methodology, guiding the phases from business understanding to deployment and continuous improvement, ensuring quality assurance throughout the process.

## Technology Stack

- **OCR Tools Used**: Google OCR and Paddle OCR for extracting data from images.
- **Data Annotation**: Images were annotated to identify the required data for extraction.
- **Data Processing**: Extracted data is converted into JSON format, leveraging the Gemini-Vision LLM.
- **Data Output**: The processed data is transformed into a digital table that can be downloaded. Users can upload a batch of images, and Gemini-Vision will provide data in JSON format. The JSON data is then converted to a Pandas DataFrame and exported to Excel.

## How to Run the Project

1. **Clone the repository** to your local machine using:
   ```bash
   git clone https://github.com/yourusername/Data-Automation-for-Solar-Power-Measurements.git
2. Navigate to the project directory:
    ```bash
   cd Data-Automation-for-Solar-Power-Measurements
3. Install the required dependencies:
    ```bash
   pip install -r requirements.txt

5. Run the main script to start the OCR and data processing:
   ```bash
    streamlit run appb.py
7. Upload Batch of Images: Users can upload a batch of images, and the system will extract data using the configured OCR tools.

8. Download Processed Data: Once the data is processed, it will be available for download in Excel format.

## Image Data Preview

![Data Preview](https://github.com/Pavun-KumarCH/Data-Automation-for-Solar-Power-Measurements/blob/main/data/data%20report-1.jpg)

## Project Deployment Showcase

You can see a live deployment of the project and its functionality below:

![Project Deployment](path/to/your/deployment/image.jpg)


## Contributing

If you wish to contribute to this project, please fork the repository and submit a pull request. For any issues or suggestions, feel free to open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

