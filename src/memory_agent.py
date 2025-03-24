from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import VectorStore
from datetime import datetime
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from typing import Dict

from src.chunk_extractor import extract_chunks
import re

class MemoryAgent:
    def __init__(self, vector_store: VectorStore):
        # Using a Local LLM
        self.use_local_llm()

        self.vector_store = vector_store
        self.system_prompt = PromptTemplate.from_template("""
            You are an AI assistant with a memory system.
            Your task is to extract and store relevant memory information from user queries.
            Analyze the following query and identify any personal or relevant information that should be remembered:

            Query: {user_query}
            
            Extract relevant details like any of these:
            - Name
            - Profession
            - Personal preferences
            - Important facts
            - Events and stories about the user

            Only output the relevant information. Each information extracted should be sepparated by a new line.
            If no relevant information is found, return "No memorable information" and no other text.
            Otherwise, return the information in a clear format.
            """)

    def use_local_llm(self):
        self.llm = OllamaLLM(
            model="deepseek-r1:14b",
            temperature=0.7
        )

    def _memory_extraction_step(
        self,
        query: str, 
        vector_store: VectorStore) -> Dict[str, str]:
        # Extract potential memory information
        formatted_prompt = self.system_prompt.format(user_query=query)

        memory_info = self.llm.invoke(formatted_prompt)

        # Parse the memory info to remove any text between <think> tags
        def remove_think_tags(text: str) -> str:
            return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        memory_info = remove_think_tags(memory_info)
        
        # If relevant information was found, store it
        if not memory_info.__contains__("No memorable information"):
            print("<-- Saving memory to db -->")
            chunks = extract_chunks(memory_info)
            for chunk in chunks:
                print(f"Chunk saved: ${chunk}")
                vector_store.add_texts(
                    texts=[chunk.page_content],
                    metadatas=[{"source": "user_query", "timestamp": str(datetime.now())}]
                )
        print("<-- End of saving memory -->")
        return {"query": query, "extracted_memory": memory_info}

    def run_memory_agent(self, query):
        """
        Runs the memory agent on a given query

        Args:
            query: The user query
        """
        answer = self._memory_extraction_step(query=query, vector_store=self.vector_store)

        print("Relevant information extracted:" + answer["extracted_memory"])
