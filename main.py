
**Important Notes:**
- Ensure all fields are populated correctly; if a field is not applicable, leave it blank.
- Do not include any text before or after the CSV content.
- Do not hallucinate PDF names; use the exact names from the Knowledge Base.
- Provide accurate page numbers and original question numbers from the PDFs.
- Ensure that the Correct Answer (English) field contains the full text of the correct option.
- Explanations should be detailed, at least 2-3 paragraphs, and explain why other options are not suitable.
- Use only information that can be directly verified from the Knowledge Base.
- If no relevant entry is found in the Knowledge Base, respond with: "Not found in knowledge text."
"""

    retry_count = 0

    while retry_count < MAX_RETRIES:
        try:
            # Create a new thread for each run
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
                    # Get the last assistant message
                    assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
                    if not assistant_messages:
                        st.error("No assistant response found.")
                        return None

                    last_message = assistant_messages[-1]
                    csv_content = last_message.content[0].text.value

                    # Process CSV content
                    df = process_csv_content(csv_content, language_param)
                    if df is not None and not df.empty:
                        return df
                    else:
                        st.warning("No valid CSV content generated. Retrying...")
                        retry_count += 1
                        break
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

def generate_questions(params, api_key):
    subjects, topics, sub_topic, selected_pdfs, keywords, question_types, num_questions, difficulty_levels, language, question_source, year_range = params

    cumulative_df = pd.DataFrame()
    total_questions = 0
    dataframe_placeholder = st.empty()

    # Adjust batch size based on total number of questions
    if num_questions >= 500:
        batch_size = 50
    elif num_questions >= 100:
        batch_size = 20
    else:
        batch_size = 5

    progress_bar = st.progress(0)
    status_placeholder = st.empty()

    while total_questions < num_questions:
        current_batch_size = min(num_questions - total_questions, batch_size)
        df = generate_questions_batch(params, api_key, current_batch_size, language)
        
        if df is not None and not df.empty:
            cumulative_df = pd.concat([cumulative_df, df], ignore_index=True)
            cumulative_df.drop_duplicates(subset=['Question Text (English)', 'Question Text (Hindi)'], inplace=True)
            total_questions = len(cumulative_df)
            
            # Update the displayed dataframe
            dataframe_placeholder.dataframe(cumulative_df)
            
            # Update progress bar
            progress = min(total_questions / num_questions, 1.0)
            progress_bar.progress(progress)
            
            status_placeholder.success(f"Total questions generated: {total_questions}/{num_questions}")
        else:
            st.warning("A batch did not return any questions. Retrying...")

    if total_questions < num_questions:
        st.warning(f"Only {total_questions} unique questions could be generated out of the requested {num_questions}.")
    else:
        st.success(f"All {num_questions} questions generated successfully.")

    return cumulative_df

def main():
    st.title("Drishti QueAI")
    st.markdown(CSS, unsafe_allow_html=True)
    api_key = st.text_input("Enter your API Key:", type="password")

    if not api_key:
        st.warning("Please enter your API key to proceed.")
        return

    if not validate_api_key(api_key):
        return

    params = create_sidebar()
    num_questions = params[6]  # Extract num_questions from params

    if st.sidebar.button("Generate Questions"):
        with st.spinner("Generating questions..."):
            cumulative_df = generate_questions(params, api_key)

        if cumulative_df is not None and not cumulative_df.empty:
            st.success(f"Generated {len(cumulative_df)} questions successfully.")

            csv_data = cumulative_df.to_csv(index=False)
            st.download_button(label="Download CSV", data=csv_data, file_name="generated_questions.csv", mime="text/csv")
        else:
            st.error("No questions were generated. Please try again.")

if __name__ == "__main__":
    main()
