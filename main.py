import streamlit as st
import pandas as pd
import io
import time
import random
from openai import OpenAI
from subject_data import SUBJECTS, TOPICS, PDF_NAMES

# Constants
ASSISTANT_ID = "asst_1cp5iEnWInbKxO05X1fEVKFC"

# Inspirational quotes
QUOTES = [
    "Every question you tackle brings you closer to success.",
    "Knowledge is the key; perseverance unlocks the door.",
    "In the journey of learning, curiosity is your best companion.",
    "Today's effort is tomorrow's excellence.",
    "Embrace the challenge; it's shaping your future.",
    "Small steps lead to big achievements in UPSC preparation.",
    "Your dedication today paves the way for tomorrow's success.",
    "Each question mastered is a step towards your goal.",
    "In the world of UPSC, consistency is the true key to success.",
    "Challenge your limits, expand your knowledge, achieve your dreams."
]

def get_random_quote():
    return random.choice(QUOTES)

def create_sidebar():
    st.sidebar.title("Question Generator")
    subject = st.sidebar.selectbox("Subject", SUBJECTS)
    topic = st.sidebar.selectbox("Topic", TOPICS.get(subject, []))
    sub_topic = st.sidebar.text_input("Sub-Topic", "General")
    
    pdf_options = PDF_NAMES.get(subject, {})
    if isinstance(pdf_options, dict):
        pdf_options = pdf_options.get(topic, [])
    selected_pdf = st.sidebar.selectbox("Select PDF", pdf_options)
    
    question_type = st.sidebar.selectbox("Question Type", ["MCQ", "Fill in the Blanks", "Short Answer", "Descriptive/Essay", "Match the Following", "True/False"])
    num_questions = st.sidebar.number_input("Number of Questions", min_value=1, max_value=50, value=5)
    difficulty = st.sidebar.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
    language = st.sidebar.selectbox("Language", ["English", "Hindi", "Both"])
    question_source = st.sidebar.selectbox("Question Source", ["Rewrite existing", "Create new"])
    
    return subject, topic, sub_topic, selected_pdf, question_type, num_questions, difficulty, language, question_source

def validate_api_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception as e:
        return False

def generate_questions(params, api_key):
    subject, topic, sub_topic, selected_pdf, question_type, num_questions, difficulty, language, question_source = params
    
    client = OpenAI(api_key=api_key)

    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Generate {num_questions} {question_type} questions for {subject} - {topic} - {sub_topic}. Use content from {selected_pdf}. Difficulty: {difficulty}. Language: {language}. Source: {question_source}. Please provide the output in CSV format with the following headers: Subject,Topic,Sub-Topic,Question Type,Question Text (English),Question Text (Hindi),Option A (English),Option B (English),Option C (English),Option D (English),Option A (Hindi),Option B (Hindi),Option C (Hindi),Option D (Hindi),Correct Answer (English),Correct Answer (Hindi),Explanation (English),Explanation (Hindi),Difficulty Level,Language,Source PDF Name,Source Page Number,Original Question Number,Year of Original Question"
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )

    while run.status != "completed":
        time.sleep(1)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    last_message = messages.data[0]
    csv_content = last_message.content[0].text.value

    return csv_content

def process_csv_content(csv_content):
    # Remove any text before the actual CSV data
    csv_start = csv_content.find("Subject,Topic,")
    if csv_start != -1:
        csv_content = csv_content[csv_start:]

    # Read CSV content
    df = pd.read_csv(io.StringIO(csv_content))
    
    # Ensure all expected columns are present
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
    
    return df[expected_columns]

def main():
    st.set_page_config(page_title="Drishti QueAI", page_icon="📚", layout="wide")
    
    # Load custom CSS
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    st.title("Drishti QueAI")

    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    api_key = st.text_input("Enter your API Key:", value=st.session_state.api_key, type="password")
    st.session_state.api_key = api_key

    if not api_key:
        st.warning("Please enter your API key to proceed.")
        st.stop()

    if not validate_api_key(api_key):
        st.error("Invalid API key. Please enter a valid API key.")
        st.stop()

    params = create_sidebar()

    if st.sidebar.button("Generate Questions"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(100):
            status_text.text(get_random_quote())
            progress_bar.progress(i + 1)
            time.sleep(0.1)

        try:
            csv_content = generate_questions(params, api_key)
            
            df = process_csv_content(csv_content)
            
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="generated_questions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()
