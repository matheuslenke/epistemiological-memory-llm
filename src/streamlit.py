import streamlit as st

def initialize_streamlit():
    st.set_page_config(
        page_title="ChatBot com Memória - Matheus Lenke e Eduardo Santos",
        page_icon=":robot:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    with st.sidebar:
        st.header("📚 Guia de utilização deste ChatBot")
        st.markdown("Em breve")
    st.session_state.messages = [{"role": "system", "content": """You are an AI assistant with an epistemological memory system. 
        Use the following pieces of retrieved context to answer the user's question, if necessary.
        Always answer the question considering the memory context that is provided, but do not be limited by it.
        You will receive a Context and a Query.
        """}]

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Let's start chatting! 👇"}]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        # Skip system messages. They do not need to be shown for the user
        if (message["role"] != "system"):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # create the bar where we can type messages
    prompt = st.chat_input("Welcome to the Epistemiological Memory LLM Agent. Please ask a question.")
    return prompt