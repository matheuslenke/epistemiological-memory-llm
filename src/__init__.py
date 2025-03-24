import os
import getpass
from dotenv import load_dotenv

import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.vectorstores import VectorStore

from src.chat_agent import ChatAgent
from src.memory_agent import MemoryAgent

# Load environment variables
load_dotenv()

# Configure Google Generative AI
if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

def initialize_streamlit():
    st.set_page_config(
        page_title="ChatBot com Memória - Matheus Lenke e Eduardo",
        page_icon=":robot:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("ChatBot com Memória - Matheus Lenke e Eduardo")

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # Skip system messages. They do not need to be shown for the user
        if (message["role"] != "system"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    for message in st.session_state.messages:
        print(message)

    # create the bar where we can type messages
    prompt = st.chat_input("Welcome to the Epistemiological Memory LLM Agent. Please ask a question.")
    return prompt


def initialize_vector_db() -> VectorStore:
    # initialize embeddings model + vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        collection_name="memory_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    return vector_store

def retrieve_similar_documents(user_prompt: str):
    # creating and invoking the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )

    docs = retriever.invoke(user_prompt)
    return docs


# Example usage
if __name__ == "__main__":
    vector_store = initialize_vector_db()

    user_prompt = initialize_streamlit()
    chat_agent = ChatAgent()
    memory_agent = MemoryAgent(vector_store=vector_store)

    if user_prompt:
        # add the message from the user (prompt) to the screen with streamlit
        with st.chat_message("user"):
            st.markdown(user_prompt)
            st.session_state.messages.append({"role": "user", "content": user_prompt})

        # 1. Analyze the user's prompt and extract relevant memory information
        memory_agent.run_memory_agent(query=user_prompt)

        # 2. Retrieve relevant documents from the vector store based on the user's prompt
        docs = retrieve_similar_documents(user_prompt)
        docs_text = "".join(d.page_content for d in docs)

        # 3. Generate a response from the llm using the user's prompt and the retrieved documents
        stream = chat_agent.chat(user_prompt, docs_text, st)
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
            st.session_state.messages.append({"role": "assistant", "content": response})
