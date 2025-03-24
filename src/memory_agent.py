from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import VectorStore
from datetime import datetime
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from typing import Dict

from src.chunk_extractor import extract_chunks

class MemoryAgent:
    def __init__(self, vector_store: VectorStore):
        self.llm = OllamaLLM(
            model="llama3.2:latest",
            temperature=0.7
        )
        self.vector_store = vector_store
        self.system_prompt = PromptTemplate.from_template("""
            You are an AI assistant with a memory system.
            Your task is to extract and store relevant memory information from user queries.
            Analyze the following query and identify any personal or relevant information that should be remembered:

            Query: {user_query}
            
            Extract relevant details like:
            - Name
            - Profession
            - Personal preferences
            - Important facts
            - Events and stories about the user

            It does not need to contain all of them, if one is present, then it is enough.
            
            If no relevant information is found, return "No memorable information".
            Otherwise, return the information in a clear format as a single paragraph.
            """)

    def _memory_extraction_step(
        self,
        query: str, 
        vector_store: VectorStore) -> Dict[str, str]:
        # Extract potential memory information
        formatted_prompt = self.system_prompt.format(user_query=query)

        memory_info = self.llm.invoke(formatted_prompt)
        
        # If relevant information was found, store it
        if memory_info != "No memorable information":
            chunks = extract_chunks(memory_info)
            for chunk in chunks:
                vector_store.add_texts(
                    texts=[chunk.page_content],
                    metadatas=[{"source": "user_query", "timestamp": str(datetime.now())}]
                )
            
        return {"query": query, "extracted_memory": memory_info}

    def run_memory_agent(self, query):
        """
        Runs the memory agent on a given query

        Args:
            query: The user query
        """
        answer = self._memory_extraction_step(query=query, vector_store=self.vector_store)

        print("Relevant information extracted:" + answer["extracted_memory"])
