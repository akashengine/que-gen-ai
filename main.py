import openai
import time
import pandas as pd
import io
import streamlit as st
from subject_data import SUBJECTS, TOPICS, PDF_NAMES

# Constants
ASSISTANT_ID = "asst_WejSQNw2pN2DRnUOXpU3vMeX"
MAX_COMPLETION_TOKENS = 16384
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 128000  # Maximum allowed tokens for context

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
st.markdown(CSS, unsafe_allow_html=True)

# Sidebar for input parameters
def create_sidebar():
    st.sidebar.title("Question Generator")

    # Subjects
    subjects = st.sidebar.multiselect("Select Subject(s)", SUBJECTS)

    # Dynamically generate topics based on selected subjects
    all_topics = []
    for subject in subjects:
        all_topics.extend(TOPICS.get(subject, []))

    # Select topics based on subjects
    topics = st.sidebar.multiselect("Select Topic(s)", list(set(all_topics)))

    # Dynamically generate sub-topics if applicable
    sub_topic = st.sidebar.selectbox("Sub-Topic", topics if topics else ["General"])

    # Dynamically generate PDF options based on selected subjects and topics
    pdf_options = []
    for subject in subjects:
        subject_pdfs = PDF_NAMES.get(subject, {})
        if isinstance(subject_pdfs, dict):
            for topic in topics:
                pdf_options.extend(subject_pdfs.get(topic, []))
        else:
            pdf_options.extend(subject_pdfs)

    selected_pdfs = st.sidebar.multiselect("Select PDF(s)", list(set(pdf_options)))

    # Keywords input
    keywords = st.sidebar.text_input("Keywords", "Enter any keywords")

    # Question types, number of questions, difficulty levels, language, and source selection
    question_types = st.sidebar.multiselect("Question Type(s)", ["MCQ", "Fill in the Blanks", "Short Answer", "Descriptive/Essay", "Match the Following", "True/False"])
    num_questions = st.sidebar.number_input("Number of Questions", min_value=1, max_value=250, value=5)
    difficulty_levels = st.sidebar.multiselect("Difficulty Level(s)", ["Easy", "Medium", "Hard"])
    language = st.sidebar.selectbox("Language", ["English", "Hindi", "Both"])
    question_source = st.sidebar.selectbox("Question Source", ["Rewrite existing", "Create new"])
    year_range = st.sidebar.slider("Year Range", 1947, 2024, (2000, 2024))

    return subjects, selected_pdfs, sub_topic, keywords, question_types, num_questions, difficulty_levels, language, question_source, year_range

# Function to validate API Key
def validate_api_key(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception as e:
        st.error("Invalid API key")
        return False

# Function to count tokens
def count_tokens(text, model="gpt-4o"):
    enc = openai.api.tokenizer.get_encoding(model)
    return len(enc.encode(text))

# Function to chunk input data if needed
def chunk_input(text, max_tokens=MAX_COMPLETION_TOKENS):
    enc = openai.api.tokenizer.get_encoding("gpt-4o")
    tokens = enc.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [" ".join(enc.decode(chunk)) for chunk in chunks]

# Generate questions with continuous runs
def generate_questions(params, api_key):
    client = openai.OpenAI(api_key=api_key)

    subjects, selected_pdfs, sub_topic, keywords, question_types, num_questions, difficulty_levels, language, question_source, year_range = params
    subjects_text = ", ".join(subjects) if subjects else "No specific subject selected"
    pdf_text = ", ".join(selected_pdfs) if selected_pdfs else "No specific PDF selected"
    question_types_text = ", ".join(question_types)
    difficulty_levels_text = ", ".join(difficulty_levels)

    total_questions = 0
    all_csv_content = ""
    
    while total_questions < num_questions:
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
        3. Format the output as a CSV.
        """
        
        prompt_tokens = count_tokens(prompt)
        if prompt_tokens >= MAX_COMPLETION_TOKENS:
            prompt_chunks = chunk_input(prompt)
            prompt = prompt_chunks[0]
        
        # Start a new thread
        thread = client.beta.threads.create()
        
        # Send message
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        # Run the assistant with the GPT-4o model
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            model=MODEL_NAME,
            max_completion_tokens=MAX_COMPLETION_TOKENS
        )

        # Polling the status until completion
        while run.status != "completed":
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

        # Fetch generated messages
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        last_message = messages.data[-1]
        csv_content = last_message.content[0].text.value
        
        if "incomplete" in csv_content:
            st.warning("The run was incomplete. Retrying for the remaining questions.")
            continue  # Retry for the remaining questions
        
        # Append generated content to the final output
        all_csv_content += csv_content
        total_questions += csv_content.count("Question Text")  # Assuming CSV format includes "Question Text" field for each question

    return all_csv_content

# Main function
def main():
    st.title("Drishti QueAI")
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
            # Display the CSV content as a DataFrame
            df = pd.read_csv(io.StringIO(csv_content))
            st.dataframe(df)
            
            # Allow downloading the generated CSV
            csv_data = df.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv_data, file_name="generated_questions.csv", mime="text/csv")

if __name__ == "__main__":
    main()
