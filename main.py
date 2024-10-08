import openai
import time
import pandas as pd
import io
import streamlit as st
import tiktoken
from subject_data import SUBJECTS, TOPICS, PDF_NAMES

# Constants
ASSISTANT_ID = "asst_WejSQNw2pN2DRnUOXpU3vMeX"
MAX_COMPLETION_TOKENS = 16384
MODEL_NAME = "gpt-4o"
MAX_TOKENS = 128000
MAX_RETRIES = 3
POLLING_INTERVAL = 5  # seconds

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

# Function to validate API Key
def validate_api_key(api_key):
    try:
        client = openai.OpenAI(api_key=api_key)
        client.models.list()
        return True
    except Exception as e:
        st.error("Invalid API key")
        return False

# Function to count tokens using tiktoken
def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Function to chunk input data if needed
def chunk_input(text, max_tokens=MAX_COMPLETION_TOKENS):
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(text)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [encoding.decode(chunk) for chunk in chunks]

# Sidebar for input parameters
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
        3. Format the output as a CSV.
        """
        
        prompt_tokens = count_tokens(prompt)
        if prompt_tokens >= MAX_COMPLETION_TOKENS:
            prompt_chunks = chunk_input(prompt)
            prompt = prompt_chunks[0]
        
        thread = client.beta.threads.create()
        
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt
        )

        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=ASSISTANT_ID,
            model=MODEL_NAME,
            max_completion_tokens=MAX_COMPLETION_TOKENS
        )

        start_time = time.time()
        max_run_time = 600  # 10 minutes in seconds

        while run.status not in ["completed", "requires_action", "failed", "cancelled", "expired"]:
            time.sleep(POLLING_INTERVAL)
            run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            
            if time.time() - start_time > max_run_time:
                st.warning("Run took too long. Retrying...")
                break

        if run.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            last_message = messages.data[0]  # Get the most recent message
            csv_content = last_message.content[0].text.value
            
            if "incomplete" in csv_content.lower():
                st.warning("The run was incomplete. Retrying for the remaining questions.")
                retry_count += 1
            else:
                all_csv_content += csv_content
                new_questions = csv_content.count("Question Text")
                total_questions += new_questions
                retry_count = 0  # Reset retry count on successful generation
                st.success(f"Generated {new_questions} questions. Total: {total_questions}/{num_questions}")
        elif run.status in ["failed", "cancelled", "expired"]:
            st.error(f"Run {run.status}. Retrying...")
            retry_count += 1
        elif run.status == "requires_action":
            st.error("Run requires action. This shouldn't happen with our current setup. Retrying...")
            retry_count += 1

        if retry_count >= MAX_RETRIES:
            st.error(f"Failed to generate questions after {MAX_RETRIES} attempts. Please try again later.")
            break

    return all_csv_content

# Main function
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
            df = pd.read_csv(io.StringIO(csv_content))
            st.dataframe(df)
            
            csv_data = df.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv_data, file_name="generated_questions.csv", mime="text/csv")

if __name__ == "__main__":
    main()
