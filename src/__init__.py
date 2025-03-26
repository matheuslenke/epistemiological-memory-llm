import os
import getpass
from dotenv import load_dotenv

import streamlit as st

from src.chat_agent import ChatAgent
from src.memory_agent import MemoryAgent
from src.embeddings import initialize_vector_db, retrieve_similar_documents
from src.streamlit import initialize_streamlit

# Load environment variables
load_dotenv()

# Configure Google Generative AI
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")
    
def main():
    """Main function that initializes and runs the chat application with memory capabilities.
    
    This function:
    1. Initializes the vector database for document storage
    2. Sets up the Streamlit interface
    3. Creates chat and memory agents
    4. Processes user input by:
        - Analyzing the prompt for memory information
        - Retrieving relevant documents
        - Filtering out non-memorable information
        - Generating an AI response using the context
    """
    vector_store = initialize_vector_db()

    user_prompt = initialize_streamlit()
    chat_agent = ChatAgent()
    memory_agent = MemoryAgent(vector_store=vector_store)

    if user_prompt:
        # 1. Add the message from the user (prompt) to the screen with streamlit
        with st.chat_message("user"):
            st.markdown(user_prompt)
            st.session_state.messages.append({"role": "user", "content": user_prompt})

        # 2. Analyze the user's prompt and extract relevant memory information
        memory_agent.run_memory_agent(query=user_prompt)

        # 3. Retrieve relevant documents from the vector store based on the user's prompt
        docs = retrieve_similar_documents(user_prompt, vector_store=vector_store)
        filtered_docs = []

        # 4. Filter documents, to make sure no "No memorable information" is provided.
        for doc in docs:
            if not doc.page_content.__contains__("No memorable information"):
                filtered_docs.append(doc)
        docs_text = "\n".join(d.page_content for d in filtered_docs)

        # 5. Generate a response from the llm using the user's prompt and the retrieved documents
        stream = chat_agent.chat(user_prompt, docs_text, st)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.write_stream(stream)
                print(response)
                st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()