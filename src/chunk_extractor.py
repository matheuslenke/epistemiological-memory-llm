from typing import List, Dict
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def extract_chunks(data: str ) -> List[Document]:
    """
    Extracts chunks from input text data using recursive character splitting.
    
    This function takes a string input and splits it into smaller chunks while maintaining context.
    The chunks are created using a recursive character splitter that:
    - Creates chunks of approximately 200 characters
    - Maintains a 10 character overlap between chunks to preserve context
    - Splits on natural boundaries like paragraphs and sentences when possible
    
    Args:
        data (str): The input text to be split into chunks
        
    Returns:
        List[Document]: A list of Document objects, where each Document contains a chunk of the original text
    """
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)

    # Create a Document object with content
    chunks = text_splitter.split_documents([Document(page_content=data)])

    # Check the chunks
    return chunks