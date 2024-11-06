import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from populate_db import get_embedding_function, clear_database, load_documents, split_documents, add_to_chroma

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Give an explanation on how to go about getting the answer. 
Do not solve the exact problem that the student has provided; instead, use variables and other values to explain the problem.
If you do provide the answer, give it followed by the string "answer is."
Also, provide tips and potential mistakes the user might make based only on the following context:

{context}

---

{question}
"""

# Initialize database and model
def initialize_db():
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    return db

def reset_database():
    clear_database()
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)

# Streamlit UI for querying
def query_database(query_text, db):
    # Search Chroma DB with similarity
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Format the prompt with context
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Generate response
    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)
    
    # Post-process response to remove direct answers if needed
    processed_response = post_process_response(response_text)
    
    # Collect sources for user reference
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {processed_response}\nSources: {sources}"
    return formatted_response

# Post-process response to avoid direct answers
def post_process_response(response):
    sentences = response.split('.')
    filtered_sentences = [
        sentence for sentence in sentences 
        if not any(keyword in sentence.lower() for keyword in ['answer is', 'solution is', 'result is'])
    ]
    processed_response = '. '.join(filtered_sentences)
    return processed_response

# Streamlit app layout
def main():
    st.title("Math 127 Chatbot")
    
    # Initialize or reset DB if needed
    if st.sidebar.button("Reset DB"):
        reset_database()
        st.sidebar.success("Database reset successfully!")
    
    # Display input box for user query
    user_query = st.text_input("Ask a question:")
    
    if "db" not in st.session_state:
        st.session_state.db = initialize_db()
        
    # Process the query
    if st.button("Submit Query") and user_query:
        response = query_database(user_query, st.session_state.db)
        st.write(response)

if __name__ == "__main__":
    main()
