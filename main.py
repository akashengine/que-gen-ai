import streamlit as st
import pandas as pd
import io
import time
import random
from openai import OpenAI
from subject_data import SUBJECTS, TOPICS, PDF_NAMES

# Constants
ASSISTANT_ID = "asst_XTavO6ea0DASnN8cvBmYVxDX"

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

# CSS styles
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

def get_random_quote():
    return random.choice(QUOTES)

def create_sidebar():
    st.sidebar.title("Question Generator")
    subjects = st.sidebar.multiselect("Select Subject(s)", SUBJECTS)
    
    all_topics = []
    for subject in subjects:
        all_topics.extend(TOPICS.get(subject, []))
    topics = st.sidebar.multiselect("Select Topic(s)", list(set(all_topics)))
    
    sub_topic = st.sidebar.text_input("Sub-Topic", "General")
    
    pdf_options = []
    for subject in subjects:
        subject_pdfs = PDF_NAMES.get(subject, {})
        if isinstance(subject_pdfs, dict):
            for topic in topics:
                pdf_options.extend(subject_pdfs.get(topic, []))
        else:
            pdf_options.extend(subject_pdfs)
    selected_pdfs = st.sidebar.multiselect("Select PDF(s)", list(set(pdf_options)))
    
    question_types = st.sidebar.multiselect("Question Type(s)", ["MCQ", "Fill in the Blanks", "Short Answer", "Descriptive/Essay", "Match the Following", "True/False"])
    num_questions = st.sidebar.number_input("Number of Questions", min_value=1, max_value=50, value=5)
    difficulty_levels = st.sidebar.multiselect("Difficulty Level(s)", ["Easy", "Medium", "Hard"])
    language = st.sidebar.selectbox("Language", ["English", "Hindi", "Both"])
    question_source = st.sidebar.selectbox("Question Source", ["Rewrite existing", "Create new"])
    year_range = st.sidebar.slider("Year Range", 1947, 2024, (2000, 2024))
    
    return subjects, topics, sub_topic, selected_pdfs, question_types, num_questions, difficulty_levels, language, question_source, year_range

def validate_api_key(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception as e:
        return False

def generate_questions(params, api_key):
    subjects, topics, sub_topic, selected_pdfs, question_types, num_questions, difficulty_levels, language, question_source, year_range = params
    
    client = OpenAI(api_key=api_key)
    subjects_text = ", ".join(subjects) if subjects else "No specific subject selected"
    topics_text = ", ".join(topics) if topics else "No specific topic selected"
    pdf_text = ", ".join(selected_pdfs) if selected_pdfs else "No specific PDF selected"
    question_types_text = ", ".join(question_types)
    difficulty_levels_text = ", ".join(difficulty_levels)

    thread = client.beta.threads.create()

    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"""
        Generate {num_questions} questions based on the following parameters:
        • Subjects: {subjects_text}
        • Topics: {topics_text}
        • Sub-Topic: {sub_topic}
        • Question Type(s): {question_types_text}
        • Difficulty Level(s): {difficulty_levels_text}
        • Language: {language}
        • Question Source: {question_source}
        • Year Range: {year_range[0]} to {year_range[1]}
        • Reference Material: {pdf_text}

        Instructions:
        1. Use the specified PDFs as reference material.
        2. For each question, use the actual question number from the referenced PDF page.
        3. Ensure the year for each question falls within the specified range.
        4. Provide answers and explanations in both English and Hindi if the language is set to "Both".
        5. Format the output as a CSV with the following headers:
           Subject,Topic,Sub-Topic,Question Type,Question Text (English),Question Text (Hindi),Option A (English),Option B (English),Option C (English),Option D (English),Option A (Hindi),Option B (Hindi),Option C (Hindi),Option D (Hindi),Correct Answer (English),Correct Answer (Hindi),Explanation (English),Explanation (Hindi),Difficulty Level,Language,Source PDF Name,Source Page Number,Original Question Number,Year of Original Question

        Important: Do not generate any text before or after the CSV content. The response should contain only the CSV data.
        """
    )

    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID
    )

    while run.status != "completed":
        time.sleep(0.5)
        run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

    messages = client.beta.threads.messages.list(thread_id=thread.id)
    last_message = messages.data[0]
    csv_content = last_message.content[0].text.value

    return csv_content

def process_csv_content(csv_content):
    # Check if the content is not found in knowledge text
    if "Not found in knowledge text" in csv_content:
        return None

    # Remove any text before the actual CSV data
    csv_start = csv_content.find("Subject,Topic,")
    if csv_start != -1:
        csv_content = csv_content[csv_start:]

    # Read CSV content
    try:
        df = pd.read_csv(io.StringIO(csv_content))
    except pd.errors.ParserError:
        st.error("Error parsing CSV data. The assistant's response may not be in the correct format.")
        return None
    
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
    
    # Apply CSS
    st.markdown(CSS, unsafe_allow_html=True)
    
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

        start_time = time.time()
        while time.time() - start_time < 3:  # Keep the loading bar for only 3 seconds
            progress_bar.progress(min(int((time.time() - start_time) / 3 * 100), 100))
            status_text.text(get_random_quote())
            time.sleep(0.5)

        try:
            csv_content = generate_questions(params, api_key)
            df = process_csv_content(csv_content)
            
            if df is None:
                st.warning("No questions could be generated based on the given parameters. The content may not be found in the knowledge text.")
            else:
                # Save generated DataFrame to session state
                st.session_state.generated_df = df
                
                # Display the DataFrame with horizontal scrolling
                st.markdown(
                    """
                    <style>
                    .stDataFrame {
                        width: 100%;
                        overflow-x: auto;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.dataframe(df)
                
                # Create CSV from DataFrame and provide download button
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
