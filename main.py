import openai
import time
import pandas as pd
import io
import streamlit as st
import tiktoken
from subject_data import SUBJECTS, TOPICS, PDF_NAMES
import csv

import concurrent.futures  # For threading

# Constants
ASSISTANT_ID = "asst_WejSQNw2pN2DRnUOXpU3vMeX"
MAX_COMPLETION_TOKENS = 16384
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 128000
MAX_RETRIES = 3
POLLING_INTERVAL = 2  # seconds
MAX_RUN_TIME = 600  # 10 minutes in seconds

# CSS Styling
CSS = """
<style>
.stApp {
    background-color: #1E1E1E;
    color: #FFFFFF;
}
.stDataFrame {
    font-size: 14px;
    width: 100%;
    overflow-x: auto;
}
.stDataFrame td {
    background-color: #2D2D2D;
    color: white;
}
.stDataFrame tr:nth-child(even) {
    background-color: #353535;
}
.stProgress .st-bo {
    background-color: #4CAF50;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
.stSelectbox>div>div, .stMultiselect>div>div {
    background-color: #2D2D2D;
    color: white;
}
.stTextInput>div>div>input {
    background-color: #2D2D2D;
    color: white;
}
</style>
"""

def validate_api_key(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception as e:
        st.error(f"Invalid API key: {str(e)}")
        return False

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def create_sidebar():
    st.sidebar.title("Question Generator")

    subjects = st.sidebar.multiselect("Select Subject(s)", SUBJECTS)

    all_topics = []
    for subject in subjects:
        all_topics.extend(TOPICS.get(subject, []))

    topics = st.sidebar.multiselect("Select Topic(s)", list(set(all_topics)))

    pdf_options = []
    for subject in subjects:
        subject_pdfs = PDF_NAMES.get(subject, {})
        if isinstance(subject_pdfs, dict):
            for topic in topics:
                pdf_options.extend(subject_pdfs.get(topic, []))
        else:
            pdf_options.extend(subject_pdfs)

    selected_pdfs = st.sidebar.multiselect("Select Reference PDF(s)", list(set(pdf_options)))
    keywords = st.sidebar.text_input("Keywords (Optional)", "")
    question_types = st.sidebar.multiselect("Question Type(s)", ["MCQ", "Fill in the Blanks", "Short Answer", "Descriptive/Essay", "Match the Following", "True/False"])
    num_questions = st.sidebar.number_input("Number of Questions", min_value=1, max_value=5000, value=5)
    difficulty_levels = st.sidebar.multiselect("Difficulty Level(s)", ["Easy", "Medium", "Hard"])
    language = st.sidebar.selectbox("Language", ["English", "Hindi", "Both"])
    question_source = st.sidebar.selectbox("Question Source", ["Rewrite existing", "Create new"])
    year_range = st.sidebar.slider("Year Range", 1947, 2024, (2000, 2024))

    sub_topic = st.sidebar.text_input("Sub-Topic (Optional)", "")

    return subjects, topics, sub_topic, selected_pdfs, keywords, question_types, num_questions, difficulty_levels, language, question_source, year_range

def process_csv_content(csv_content, language):
    if "Not found in knowledge text" in csv_content or not csv_content.strip():
        return None

    # Split the content into lines
    lines = csv_content.strip().split('\n')

    # Identify headers and data lines
    header_line = None
    data_lines = []

    for line in lines:
        if not header_line and ("Subject" in line and "Topic" in line):
            header_line = line
        else:
            data_lines.append(line)

    if not header_line:
        st.error("CSV header is missing in the assistant's response.")
        return None

    processed_content = header_line + '\n' + '\n'.join(data_lines)

    try:
        df = pd.read_csv(io.StringIO(processed_content), skipinitialspace=True)
    except pd.errors.ParserError as e:
        st.error(f"Error parsing CSV data: {e}")
        return None

    expected_columns = [
        "Subject", "Topic", "Sub-Topic", "Question Type", "Question Text (English)", "Question Text (Hindi)",
        "Option A (English)", "Option B (English)", "Option C (English)", "Option D (English)",
        "Option A (Hindi)", "Option B (Hindi)", "Option C (Hindi)", "Option D (Hindi)",
        "Correct Answer (English)", "Correct Answer (Hindi)", "Explanation (English)", "Explanation (Hindi)",
        "Difficulty Level", "Language", "Source PDF Name", "Source Page Number", "Original Question Number", "Year of Original Question"
    ]

    # Ensure all expected columns are present
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""

    # Reorder columns to match expected order
    df = df[expected_columns]

    if language == "Hindi":
        columns_to_show = [col for col in df.columns if "Hindi" in col or col in ["Subject", "Topic", "Sub-Topic", "Question Type", "Difficulty Level", "Language", "Source PDF Name", "Source Page Number", "Original Question Number", "Year of Original Question"]]
    elif language == "English":
        columns_to_show = [col for col in df.columns if "Hindi" not in col]
    else:  # Both
        columns_to_show = df.columns

    return df[columns_to_show]

def generate_questions_batch(params, api_key, batch_size, language):
    client = openai.OpenAI(api_key=api_key)
    subjects, topics, sub_topic, selected_pdfs, keywords, question_types, num_questions, difficulty_levels, language_param, question_source, year_range = params

    subjects_text = ", ".join(subjects) if subjects else "No specific subject selected"
    topics_text = ", ".join(topics) if topics else "No specific topic selected"
    sub_topic_text = sub_topic if sub_topic else "No specific sub-topic selected"
    pdf_text = ", ".join(selected_pdfs) if selected_pdfs else "All available PDFs"
    question_types_text = ", ".join(question_types) if question_types else "All question types"
    difficulty_levels_text = ", ".join(difficulty_levels) if difficulty_levels else "All difficulty levels"

    prompt = f"""
You are an advanced AI assistant designed to generate high-quality exam questions for UPSC and other public service commission exams in India. You will use a comprehensive Knowledge Base (KB) of Previous Year Questions (PYQs) provided in the attached PDF files. Your goal is to emulate expert examiners' methods, ensuring precision, consistency, and adherence to the requested specifications.

# Input Parameters:
- **Subject(s):** {subjects_text}
- **Topic(s):** {topics_text}
- **Sub-Topic:** {sub_topic_text}
- **Keywords:** {keywords}
- **Question Type(s):** {question_types_text}
- **Number of Questions:** {batch_size}
- **Difficulty Level(s):** {difficulty_levels_text}
- **Language:** {language_param}
- **Question Source:** {question_source}
- **Year Range:** {year_range[0]} to {year_range[1]}
- **Reference Material:** {pdf_text}

# Instructions:

Follow all the detailed steps provided previously to generate high-quality exam questions. Ensure that you strictly adhere to the output format specified, which is a CSV-friendly format with the following headers:

