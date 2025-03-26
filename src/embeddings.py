from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.vectorstores import VectorStore

def initialize_vector_db() -> VectorStore:
    """
    Initializes and returns a vector database using Chroma with Ollama embeddings.
    
    This function sets up a vector store that can be used to store and retrieve 
    document embeddings. It uses the nomic-embed-text model from Ollama for 
    generating embeddings and Chroma as the vector database.
    
    Returns:
        VectorStore: A configured Chroma vector store instance ready for use
    """
    # initialize embeddings model + vector store
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = Chroma(
        collection_name="memory_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
    )
    return vector_store

def retrieve_similar_documents(user_prompt: str, vector_store: VectorStore):
    # creating and invoking the retriever
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.1},
    )

    docs = retriever.invoke(user_prompt)
    return docs
