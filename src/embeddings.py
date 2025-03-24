import os
import getpass
from langchain.embeddings import OpenAIEmbeddings
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def generate_embeddings_from_chunks(chunks):

    # Use getpass to securely input the API key
    api_key = getpass.getpass('Enter your OpenAI API key: ')

    # Set the API key as an environment variable
    os.environ['OPENAI_API_KEY'] = api_key

    # Initialize OpenAI embeddings
    openai_embeddings = OpenAIEmbeddings()

    # Extract the page content from the Document objects
    texts = [chunk.page_content for chunk in chunks]

    # Convert the texts into embeddings
    embeddings = openai_embeddings.embed_documents(texts)
    return embeddings

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