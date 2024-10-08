import openai
import time
import pandas as pd
import io
import streamlit as st
import tiktoken
from subject_data import SUBJECTS, TOPICS, PDF_NAMES
import csv
import concurrent.futures

# Constants
ASSISTANT_ID = "asst_WejSQNw2pN2DRnUOXpU3vMeX"
MAX_COMPLETION_TOKENS = 16384
MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 128000
MAX_RETRIES = 3
POLLING_INTERVAL = 2  # seconds
MAX_RUN_TIME = 600  # 10 minutes in seconds
MAX_QUESTIONS_PER_BATCH = 20
MAX_PARALLEL_REQUESTS = 5

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
        elif line.strip():  # Only add non-empty lines
            data_lines.append(line)

    if not header_line:
        st.error("CSV header is missing in the assistant's response.")
        return None

    expected_columns = [
        "Subject", "Topic", "Sub-Topic", "Question Type", "Question Text (English)", "Question Text (Hindi)",
        "Option A (English)", "Option B (English)", "Option C (English)", "Option D (English)",
        "Option A (Hindi)", "Option B (Hindi)", "Option C (Hindi)", "Option D (Hindi)",
        "Correct Answer (English)", "Correct Answer (Hindi)", "Explanation (English)", "Explanation (Hindi)",
        "Difficulty Level", "Language", "Source PDF Name", "Source Page Number", "Original Question Number", "Year of Original Question"
    ]

    # Process the CSV line by line
    rows = []
    for line in data_lines:
        fields = line.split(',')
        row_dict = {col: 'N/A' for col in expected_columns}
        for i, field in enumerate(fields):
            if i < len(expected_columns):
                row_dict[expected_columns[i]] = field.strip()
        rows.append(row_dict)

    df = pd.DataFrame(rows)

    # Fill NaN values with 'N/A'
    df = df.fillna('N/A')

    # Ensure numeric columns are of the correct type
    numeric_columns = ["Source Page Number", "Original Question Number", "Year of Original Question"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    # Ensure all text columns are of string type
    text_columns = [col for col in df.columns if col not in numeric_columns]
    for col in text_columns:
        df[col] = df[col].astype(str)

    # Handle language-specific columns
    if language == "Hindi":
        columns_to_show = [col for col in df.columns if "Hindi" in col or col in ["Subject", "Topic", "Sub-Topic", "Question Type", "Difficulty Level", "Language", "Source PDF Name", "Source Page Number", "Original Question Number", "Year of Original Question"]]
    elif language == "English":
        columns_to_show = [col for col in df.columns if "Hindi" not in col]
    else:  # Both
        columns_to_show = df.columns

    # Validate that each row has a question text and at least two options
    invalid_rows = df[
        ((df["Question Text (English)"] == 'N/A') & (df["Question Text (Hindi)"] == 'N/A')) |
        (((df["Option A (English)"] == 'N/A') & (df["Option B (English)"] == 'N/A')) &
         ((df["Option A (Hindi)"] == 'N/A') & (df["Option B (Hindi)"] == 'N/A')))
    ]
    if not invalid_rows.empty:
        st.warning(f"Found {len(invalid_rows)} rows with missing question text or insufficient options. These rows will be kept but marked as potentially invalid.")

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

"Subject", "Topic", "Sub-Topic", "Question Type", "Question Text (English)", "Question Text (Hindi)", "Option A (English)", "Option B (English)", "Option C (English)", "Option D (English)", "Option A (Hindi)", "Option B (Hindi)", "Option C (Hindi)", "Option D (Hindi)", "Correct Answer (English)", "Correct Answer (Hindi)", "Explanation (English)", "Explanation (Hindi)", "Difficulty Level", "Language", "Source PDF Name", "Source Page Number", "Original Question Number", "Year of Original Question"

**Important Notes:**
- Ensure all fields are populated correctly; if a field is not applicable, leave it blank.
- Do not include any text before or after the CSV content.
- **For any missing values**, leave that column blank (instead of skipping it).
- **Ensure that every column is properly mapped** in the exact sequence specified in the header.
- Ensure that the Correct Answer (English) field contains the full text of the correct option.
- Explanations should be detailed, at least 2-3 paragraphs, and explain why other options are not suitable.
- Use only information that can be directly verified from the Knowledge Base.
- If no relevant entry is found in the Knowledge Base, respond with: "Not found in knowledge text."
"""

    retry_count = 0

    while retry_count < MAX_RETRIES:
        try:
            thread = client.beta.threads.create()
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=prompt
            )
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=ASSISTANT_ID,
                model=MODEL_NAME
            )

            start_time = time.time()

            while True:
                time.sleep(POLLING_INTERVAL)
                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

                if run.status == "completed":
                    messages = client.beta.threads.messages.list(thread_id=thread.id)
                    assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
                    if not assistant_messages:
                        st.error("No assistant response found.")
                        return None

                    last_message = assistant_messages[-1]
                    csv_content = last_message.content[0].text.value
                    return csv_content  # Return the raw CSV content

                elif run.status in ["failed", "cancelled", "expired"]:
                    st.error(f"Run {run.status}. Error: {run.last_error}. Retrying...")
                    retry_count += 1
                    break
                elif run.status == "requires_action":
                    st.error("Run requires action. Retrying...")
                    retry_count += 1
                    break

                if time.time() - start_time > MAX_RUN_TIME:
                    st.warning("Run took too long. Cancelling and retrying...")
                    client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
                    retry_count += 1
                    break

        except openai.APIError as e:
            st.error(f"OpenAI API error: {str(e)}")
            retry_count += 1
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            retry_count += 1

    st.error(f"Failed to generate questions after {MAX_RETRIES} attempts.")
    return None

def generate_questions_parallel(params, api_key, num_questions, language):
    QUESTIONS_PER_BATCH = 10
    batches = []
    remaining_questions = num_questions
    while remaining_questions > 0:
        batch_size = min(remaining_questions, QUESTIONS_PER_BATCH)
        batches.append(batch_size)
        remaining_questions -= batch_size

    all_csv_content = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as executor:
        future_to_batch = {executor.submit(generate_questions_batch, params, api_key, batch_size, language): batch_size for batch_size in batches}
        completed_questions = 0

        for future in concurrent.futures.as_completed(future_to_batch):
            batch_size = future_to_batch[future]
            try:
                csv_content = future.result()
                if csv_content:
                    all_csv_content.append(csv_content)
                    completed_questions += batch_size
                    progress = completed_questions / num_questions
                    progress_bar.progress(progress)
                    status_text.text(f"Generated {completed_questions}/{num_questions} questions")
                else:
                    st.warning(f"Failed to generate a batch of {batch_size} questions.")
            except Exception as exc:
                st.error(f"An error occurred while generating a batch of {batch_size} questions: {str(exc)}")

    # Combine all CSV content
    combined_csv_content = '\n'.join(all_csv_content)

    # Remove duplicate headers
    lines = combined_csv_content.split('\n')
    header = lines[0]
    data_lines = [line for line in lines[1:] if line.strip()]  # Remove empty lines
    final_csv_content = header + '\n' + '\n'.join(data_lines)

    return final_csv_content

def main():
    st.title("Drishti QueAI")
    st.markdown(CSS, unsafe_allow_html=True)

    # Initialize session state
    if 'csv_content' not in st.session_state:
        st.session_state.csv_content = None
    if 'params' not in st.session_state:
        st.session_state.params = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None

    api_key = st.text_input("Enter your API Key:", type="password")

    if not api_key:
        st.warning("Please enter your API key to proceed.")
        return

    if not validate_api_key(api_key):
        return

    params = create_sidebar()
    st.session_state.params = params
    num_questions = params[6]  # Extract num_questions from params

    if st.button("Generate Questions"):
        with st.spinner("Generating questions..."):
            try:
                csv_content = generate_questions_parallel(params, api_key, num_questions, params[8])  # params[8] is language
                if csv_content:
                    st.session_state.csv_content = csv_content
                    st.session_state.processed_df = None  # Reset processed dataframe
                    st.success(f"Generated {num_questions} questions successfully.")
                else:
                    st.error("No questions were generated. Please try again.")
            except Exception as e:
                st.error(f"An error occurred while generating questions: {str(e)}")

    # Always display CSV content if available
    if st.session_state.csv_content:
        st.subheader("Generated CSV Content")
        st.text_area("Raw CSV", st.session_state.csv_content, height=200)
        
        # Provide option to download raw CSV
        st.download_button(
            label="Download Raw CSV",
            data=st.session_state.csv_content,
            file_name="raw_generated_questions.csv",
            mime="text/csv"
        )

        if st.button("Process CSV"):
            with st.spinner("Processing CSV..."):
                try:
                    df = process_csv_content(st.session_state.csv_content, st.session_state.params[8])  # params[8] is language
                    if df is not None and not df.empty:
                        st.session_state.processed_df = df
                        st.success(f"Processed {len(df)} questions successfully.")
                        st.info(f"Columns in processed data: {', '.join(df.columns)}")
                        st.info(f"Number of rows with all 'N/A' values: {df.eq('N/A').all(axis=1).sum()}")
                    else:
                        st.error("Failed to process CSV content. Please check the format.")
                except Exception as e:
                    st.error(f"An error occurred while processing the CSV: {str(e)}")
                    st.text("Error details:")
                    st.exception(e)

    # Display processed dataframe if available
    if st.session_state.processed_df is not None:
        st.subheader("Processed Questions")
        st.dataframe(st.session_state.processed_df)
        csv_data = st.session_state.processed_df.to_csv(index=False)
        st.download_button(
            label="Download Processed CSV",
            data=csv_data,
            file_name="processed_generated_questions.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
