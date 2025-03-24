from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def extract_chunks(data: str) -> List[Document]:
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=170, chunk_overlap=0)

    # Create a Document object with content
    chunks = text_splitter.split_documents([Document(page_content=data)])

    # Check the chunks
    return chunks