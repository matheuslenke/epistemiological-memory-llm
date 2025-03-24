from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from typing import Iterator

class ChatAgent():
    '''
    Chat agent that uses the Ollama LLM to generate responses. This llm agent
    uses the vector database to retrieve relevant context for the prompt. The 
    main idea is to the llm be able to use memorized information about the user.
    '''
    def __init__(self):
        self.chat_history = []
        self.llm = OllamaLLM(
            model="llama3.2:latest",
            temperature=0.7
        )
        self.system_prompt = PromptTemplate.from_template("""You are an AI assistant with an epistemological memory system. 
        Use the following pieces of retrieved context to answer the user's question, if necessary.
        Always answer the question considering the memory context that is provided, but do not be limited by it.
        
        Context: {context}
        
        Query: {query}
        
        Answer:""")

    def chat(self, prompt, context, st) -> Iterator[str]:
        self.chat_history.append(prompt)

        # Populate the system prompt with the retrieved context
        system_prompt_fmt = self.system_prompt.format(context=context, query=prompt)

        print("-- SYS PROMPT --")
        print(system_prompt_fmt)

        # adding the system prompt to the message history
        st.session_state.messages.append({"role": "system", "content": system_prompt_fmt})

        # invoking the llm
        stream = self.llm.stream(st.session_state.messages)
        return stream

