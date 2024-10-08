import streamlit as st
from openai import OpenAI
from typing_extensions import override
from openai import AssistantEventHandler

class EventHandler(AssistantEventHandler):
    def __init__(self):
        self.generated_rows = []
        self.current_row = []

    @override
    def on_text_created(self, text) -> None:
        if text.strip():
            self.current_row.append(text.strip())
        if len(self.current_row) == 23:  # Number of columns in our CSV
            self.generated_rows.append(",".join(self.current_row))
            self.current_row = []
            st.text_area("Generated Questions:", value="\n".join(self.generated_rows), height=400, key=f"generated_questions_{len(self.generated_rows)}")
            st.experimental_rerun()

    @override
    def on_message_done(self, message) -> None:
        if self.current_row:
            self.generated_rows.append(",".join(self.current_row))
        st.text_area("Final Output:", value="\n".join(self.generated_rows), height=400, key="final_output")
        st.experimental_rerun()

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
        2. For each question, use the actual question number and page number from the referenced PDF.
        3. Ensure the year for each question falls within the specified range.
        4. If the language is set to "Hindi", provide all text (questions, options, answers, explanations) in Hindi only.
        5. If the language is set to "English", provide all text in English only.
        6. If the language is set to "Both", provide all text in both English and Hindi.
        7. Format the output as a CSV with the following headers:
           Subject,Topic,Sub-Topic,Question Type,Question Text (English),Question Text (Hindi),Option A (English),Option B (English),Option C (English),Option D (English),Option A (Hindi),Option B (Hindi),Option C (Hindi),Option D (Hindi),Correct Answer (English),Correct Answer (Hindi),Explanation (English),Explanation (Hindi),Difficulty Level,Language,Source PDF Name,Source Page Number,Original Question Number,Year of Original Question

        Important: Generate the questions one by one, streaming each row as it's created. Do not generate any text before or after the CSV content. The response should contain only the CSV data.
        """
    )

    event_handler = EventHandler()

    with client.beta.threads.runs.create_and_stream(
        thread_id=thread.id,
        assistant_id=ASSISTANT_ID,
        instructions="Stream the generated questions row by row.",
        event_handler=event_handler
    ) as stream:
        stream.until_done()

    return event_handler.generated_rows

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
            generated_rows = generate_questions(params, api_key)
            st.success(f"Generated {len(generated_rows)} questions successfully!")
            
            # Display the generated questions in a DataFrame
            df = pd.read_csv(io.StringIO("\n".join(generated_rows)))
            st.dataframe(df)
            
            # Provide a download button for the CSV
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
