import openai
import time
import pandas as pd
import io
import streamlit as st
import tiktoken
from subject_data import SUBJECTS, TOPICS, PDF_NAMES
import csv

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
        st.error("Invalid API key")
        return False

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def chunk_input(text, max_tokens=MAX_COMPLETION_TOKENS):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [encoding.decode(chunk) for chunk in chunks]

def create_sidebar():
    st.sidebar.title("Question Generator")

    subjects = st.sidebar.multiselect("Select Subject(s)", SUBJECTS)

    all_topics = []
    for subject in subjects:
        all_topics.extend(TOPICS.get(subject, []))

    topics = st.sidebar.multiselect("Select Topic(s)", list(set(all_topics)))
    sub_topic = st.sidebar.selectbox("Sub-Topic", topics if topics else ["General"])

    pdf_options = []
    for subject in subjects:
        subject_pdfs = PDF_NAMES.get(subject, {})
        if isinstance(subject_pdfs, dict):
            for topic in topics:
                pdf_options.extend(subject_pdfs.get(topic, []))
        else:
            pdf_options.extend(subject_pdfs)

    selected_pdfs = st.sidebar.multiselect("Select PDF(s)", list(set(pdf_options)))
    keywords = st.sidebar.text_input("Keywords", "Enter any keywords")
    question_types = st.sidebar.multiselect("Question Type(s)", ["MCQ", "Fill in the Blanks", "Short Answer", "Descriptive/Essay", "Match the Following", "True/False"])
    num_questions = st.sidebar.number_input("Number of Questions", min_value=1, max_value=250, value=5)
    difficulty_levels = st.sidebar.multiselect("Difficulty Level(s)", ["Easy", "Medium", "Hard"])
    language = st.sidebar.selectbox("Language", ["English", "Hindi", "Both"])
    question_source = st.sidebar.selectbox("Question Source", ["Rewrite existing", "Create new"])
    year_range = st.sidebar.slider("Year Range", 1947, 2024, (2000, 2024))

    return subjects, selected_pdfs, sub_topic, keywords, question_types, num_questions, difficulty_levels, language, question_source, year_range

def process_csv_content(csv_content, language):
    if "Not found in knowledge text" in csv_content:
        return None

    csv_start = csv_content.find("Subject,Topic,")
    if csv_start != -1:
        csv_content = csv_content[csv_start:]

    try:
        df = pd.read_csv(io.StringIO(csv_content))
    except pd.errors.ParserError:
        st.error("Error parsing CSV data. The assistant's response may not be in the correct format.")
        return None
    
    expected_columns = [
        "Subject", "Topic", "Sub-Topic", "Question Type", "Question Text (English)", 
        "Question Text (Hindi)", "Option A (English)", "Option B (English)", 
        "Option C (English)", "Option D (English)", "Option A (Hindi)", 
        "Option B (Hindi)", "Option C (Hindi)", "Option D (Hindi)", 
        "Correct Answer (English)", "Correct Answer (Hindi)", "Explanation (English)", 
        "Explanation (Hindi)", "Difficulty Level", "Language", "Source PDF Name", 
        "Source Page Number", "Original Question Number", "Year of Original Question"
    ]
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""
    
    if language == "Hindi":
        columns_to_show = [col for col in df.columns if "Hindi" in col or col not in ["Question Text (English)", "Option A (English)", "Option B (English)", "Option C (English)", "Option D (English)", "Correct Answer (English)", "Explanation (English)"]]
    elif language == "English":
        columns_to_show = [col for col in df.columns if "Hindi" not in col]
    else:  # Both
        columns_to_show = df.columns

    return df[columns_to_show]

def generate_questions(params, api_key):
    client = openai.OpenAI(api_key=api_key)

    subjects, selected_pdfs, sub_topic, keywords, question_types, num_questions, difficulty_levels, language, question_source, year_range = params
    subjects_text = ", ".join(subjects) if subjects else "No specific subject selected"
    pdf_text = ", ".join(selected_pdfs) if selected_pdfs else "No specific PDF selected"
    question_types_text = ", ".join(question_types)
    difficulty_levels_text = ", ".join(difficulty_levels)

    total_questions = 0
    all_csv_content = ""
    retry_count = 0
    
    while total_questions < num_questions and retry_count < MAX_RETRIES:
        remaining_questions = num_questions - total_questions
        prompt = f"""
        Generate {remaining_questions} questions based on the following parameters:
        • Subjects: {subjects_text}
        • Sub-Topic: {sub_topic}
        • Keywords: {keywords}
        • Question Type(s): {question_types_text}
        • Difficulty Level(s): {difficulty_levels_text}
        • Language: {language}
        • Question Source: {question_source}
        • Year Range: {year_range[0]} to {year_range[1]}
        • Reference Material: {pdf_text}

        Instructions:
        1. Use the specified PDFs as reference material.
        2. For each question, use the actual question number and page number from the referenced PDF.
        3. Format the output as a CSV with the following columns:
           Subject,Topic,Sub-Topic,Question Type,Question Text (English),Question Text (Hindi),Option A (English),Option B (English),Option C (English),Option D (English),Option A (Hindi),Option B (Hindi),Option C (Hindi),Option D (Hindi),Correct Answer (English),Correct Answer (Hindi),Explanation (English),Explanation (Hindi),Difficulty Level,Language,Source PDF Name,Source Page Number,Original Question Number,Year of Original Question
        4. Ensure each row is properly formatted as CSV, with values separated by commas and enclosed in double quotes if necessary.
        5. Make sure to fill in all columns, especially the Correct Answer (English) column.
        6. The Correct Answer (English) should be the full text of the correct option, not just the letter.
        7. Ensure that the Difficulty Level matches the requested level(s).
        8. The Year of Original Question should be within the specified year range.
        9. Do not include a header row in the CSV output.
        """
        
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
                    last_message = messages.data[0]  # Get the most recent message
                    csv_content = last_message.content[0].text.value
                    
                    # Process CSV content
                    df = process_csv_content(csv_content, language)
                    if df is not None and not df.empty:
                        all_csv_content += csv_content + "\n"  # Add newline to separate batches
                        new_questions = len(df)
                        total_questions += new_questions
                        retry_count = 0  # Reset retry count on successful generation
                        st.success(f"Generated {new_questions} questions. Total: {total_questions}/{num_questions}")
                    else:
                        st.warning("No valid CSV content generated. Retrying...")
                        retry_count += 1
                    break
                elif run.status in ["failed", "cancelled", "expired"]:
                    st.error(f"Run {run.status}. Error: {run.last_error}. Retrying...")
                    retry_count += 1
                    break
                elif run.status == "requires_action":
                    st.error("Run requires action. This shouldn't happen with our current setup. Retrying...")
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

        if retry_count >= MAX_RETRIES:
            st.error(f"Failed to generate questions after {MAX_RETRIES} attempts. Please try again later.")
            break

    return all_csv_content

def main():
    st.title("Drishti QueAI")
    st.markdown(CSS, unsafe_allow_html=True)
    api_key = st.text_input("Enter your API Key:", type="password")

    if not api_key:
        st.warning("Please enter your API key to proceed.")
        return

    if not validate_api_key(api_key):
        st.error("Invalid API key. Please enter a valid API key.")
        return

    params = create_sidebar()

    if st.sidebar.button("Generate Questions"):
        csv_content = generate_questions(params, api_key)
        
        if csv_content:
            # Add header row to the CSV content
            header = "Subject,Topic,Sub-Topic,Question Type,Question Text (English),Question Text (Hindi),Option A (English),Option B (English),Option C (English),Option D (English),Option A (Hindi),Option B (Hindi),Option C (Hindi),Option D (Hindi),Correct Answer (English),Correct Answer (Hindi),Explanation (English),Explanation (Hindi),Difficulty Level,Language,Source PDF Name,Source Page Number,Original Question Number,Year of Original Question\n"
            csv_content_with_header = header + csv_content
            
            # Process the entire CSV content
            df = process_csv_content(csv_content_with_header, params[7])  # params[7] should be the language parameter
            
            if df is not None and not df.empty:
                st.dataframe(df)
                
                csv_data = df.to_csv(index=False)
                st.download_button(label="Download CSV", data=csv_data, file_name="generated_questions.csv", mime="text/csv")
            else:
                st.error("Failed to process the generated questions. Please try again.")
        else:
            st.error("No questions were generated. Please try again.")

if __name__ == "__main__":
    main()
