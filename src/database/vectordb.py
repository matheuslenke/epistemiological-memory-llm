import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Path to the persistent vector database
PERSISTENT_DIR = "./vector_db"

# Initialize embeddings model
embeddings = OllamaEmbeddings(model="bge-m3")

# Function to initialize or load the vector database
def initialize_vector_db():
    # Create vector_db directory if it doesn't exist
    if not os.path.exists(PERSISTENT_DIR):
        os.makedirs(PERSISTENT_DIR)
        return Chroma(embedding_function=embeddings, persist_directory=PERSISTENT_DIR)
    else:
        # Load existing vector database
        return Chroma(embedding_function=embeddings, persist_directory=PERSISTENT_DIR)

# Function to save a new response to the vector database
def save_response_to_vectordb(query, response):
    # Create a document from the query and response
    content = f"Query: {query}\nResponse: {response}"
    
    # Add to vector database
    vectordb = Chroma(embedding_function=embeddings, persist_directory=PERSISTENT_DIR)
    vectordb.add_texts([content])
    # vectordb.persist()
    print(f"Saved new interaction to vector database")
