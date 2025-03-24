from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.vectorstores import VectorStore

def initialize_vector_db() -> VectorStore:
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


# def visualize_embeddings(chunks, embeddings):
#     # Reduce dimensions to 3D
#     pca = PCA(n_components=3)

#     # Convert the list of lists to a NumPy array
#     embeddings_array = np.array(embeddings)
#     reduced_embeddings = pca.fit_transform(embeddings_array)

#     # Plotting
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2])

#     # Annotate the points (optional - can be cluttered in 3D)
#     for i, chunk in enumerate(chunks):
#         ax.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], reduced_embeddings[i, 2], f'Chunk {i+1}', fontdict={'fontsize': 12})

#     plt.title('3D Visualization of Text Chunk Embeddings')
#     ax.set_xlabel('Component 1')
#     ax.set_ylabel('Component 2')
#     ax.set_zlabel('Component 3')
#     plt.grid(True)
#     plt.show()