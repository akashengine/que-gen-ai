import streamlit as st
import pandas as pd
import io
import random
from subject_data import SUBJECTS, TOPICS, SUB_TOPICS, PDF_NAMES
from openai import OpenAI

# Constants
ASSISTANT_ID = "asst_1cp5iEnWInbKxO05X1fEVKFC"

# UPSC-related quotes
QUOTES = [
    "Success is not final, failure is not fatal: it is the courage to continue that counts.",
    "The journey of a thousand miles begins with one step.",
    "Your preparation today determines your success tomorrow.",
    "UPSC: Where knowledge meets dedication.",
    "In the world of UPSC, persistence is the key to success."
]

def get_random_quote():
    return random.choice(QUOTES)

def validate_api_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()  # This will raise an error if the API key is invalid
        return True
    except Exception as e:
        return False

def create_sidebar():
    st.sidebar.title("Question Generator")
    subject = st.sidebar.selectbox("Subject", SUBJECTS)
    topic = st.sidebar.selectbox("Topic", TOPICS.get(subject, []))
    sub_topic = st.sidebar.selectbox("Sub-Topic", SUB_TOPICS.get(topic, ["General"]))
    question_type = st.sidebar.selectbox("Question Type", ["MCQ", "Fill in the Blanks", "Short Answer", "Descriptive/Essay", "Match the Following", "True/False"])
    num_questions = st.sidebar.number_input("Number of Questions", min_value=1, max_value=50, value=5)
    difficulty = st.sidebar.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
    language = st.sidebar.selectbox("Language", ["English", "Hindi", "Both"])
    question_source = st.sidebar.selectbox("Question Source", ["Rewrite existing", "Create new"])
    
    return subject, topic, sub_topic, question_type, num_questions, difficulty, language, question_source

def process_assistant_response(response):
    # Convert the CSV string to a pandas DataFrame
    df = pd.read_csv(io.StringIO(response))
    return df

def generate_questions(params, api_key):
    subject, topic, sub_topic, question_type, num_questions, difficulty, language, question_source = params
    
    client = OpenAI(api_key=api_key)

    # Create a new thread
    thread = client.beta.threads.create()

    # Add a message to the thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"Generate {num_questions} {question_type} questions for {subject} - {topic} - {sub_topic}. Difficulty: {difficulty}. Language: {language}. Source: {question_source}."
    )

    # Run the assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )

    # Wait for the run to complete
    while run.status != "completed":
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
        st.write(f"Run status: {run.status}")

    # Retrieve the messages
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    # Get the last message (which should be the assistant's response)
    last_message = messages.data[0]
    
    # Extract the CSV content from the message
    csv_content = last_message.content[0].text.value

    return csv_content

def main():
    st.set_page_config(page_title="UPSC Question Generator", page_icon="📚", layout="wide")
    st.title("UPSC Question Generator")

    # API Key input
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

    # Sidebar for input parameters
    params = create_sidebar()

    # Main content area
    if st.sidebar.button("Generate Questions"):
        with st.spinner("Generating questions... " + get_random_quote()):
            try:
                csv_content = generate_questions(params, api_key)
                
                df = process_assistant_response(csv_content)
                st.dataframe(df)
                
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="generated_questions.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
