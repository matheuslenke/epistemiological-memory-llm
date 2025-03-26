# Epistemiological Memory LLM

A LangChain-based application that implements epistemiological memory capabilities in AI conversations, in order to experiment how to make Large Language Models to remember information. The system analyzes user queries, extracts and stores relevant information, and uses semantic search to provide context-aware responses.

## Authors

This work was developed by:

- [Matheus Lenke Coutinho](https://www.linkedin.com/in/mlcoutinho/)
- [Eduardo Santos](https://www.linkedin.com/in/eduardo-santos-5410081a/)

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Technical Stack](#technical-stack)
- [Local LLMS needed](#local-llms-needed)
- [Setup](#setup)
- [Running the Application](#running-the-application)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a memory system that allows an AI to build and maintain knowledge about users through conversations. It uses vector embeddings to store and retrieve relevant information, creating a more personalized and context-aware conversation experience.

In order to download the `Report` telling about the foundation of the project, go to folder `docs/` and download the file `report.pdf`.

## Architecture

The project consists of two main components:

### Memory Agent

The Memory Agent is responsible for extracting and storing relevant information from user queries:

- Uses a Local LLM (deepseek-r1:14b) to analyze user queries
- Extracts important information such as:
  - Names
  - Professions
  - Personal preferences
  - Important facts
  - Events and stories
- Stores extracted information in a vector database for future reference

### Chat Agent

The Chat Agent handles the conversation flow and leverages stored memories:

- Uses Google's Gemini LLM for generating responses
- Retrieves relevant context from the vector database
- Incorporates retrieved memories into the conversation
- Maintains chat history for coherent interactions

## Technical Stack

- **Python**: Programming language
- **Poetry**: Dependency management for Python projects
- **LangChain**: Framework for building LLM applications
- **Vector Store**: For semantic storage and retrieval of memories. `ChromaDB` is used in this project.
- **LLMs**:
  - Ollama (deepseek-r1:14b) for memory extraction and chat responses
  - Nomic's `nomic-embed-text` for embeddings
- **Streamlit**: Web interface for interaction

## Local LLMS needed

In order to run this project, you need to have the following LLMs installed in `ollama`:

- deepseek-r1:14b
- nomic-embed-text

if you want to use a different LLM, you need to modify the `memory_agent.py` file and the `chat_agent.py` file.

## Setup

1. Clone the repository
2. Follow [these instructions](https://python-poetry.org/docs/#installing-with-the-official-installer) to install poetry
3. Install dependencies using Poetry:

    ```bash
    poetry install
    ```

4. Copy `.env.example` to `.env` and configure your environment variables

## Running the Application

To start the application, run:

```bash
poetry run streamlit run src/__init__.py
```

## Usage

The application works in the following way:

1. **Memory Extraction**:
   - User inputs a query
   - Memory Agent analyzes the query for relevant information
   - Extracted information is stored in the vector database

2. **Contextual Responses**:
   - When user makes a query, the system performs semantic search
   - Relevant memories are retrieved from the database
   - Chat Agent generates responses using the context

## Project Structure

```none
src/
├── chat_agent.py      # Handles conversation and response generation
├── memory_agent.py    # Extracts and stores memory information
├── chunk_extractor.py # Handles text chunking for storage
├── embeddings.py      # Manages vector embeddings
└── database/         # Vector database management
```

## Dependencies

Key dependencies include:

- langchain
- langchain-google-genai
- langchain-ollama
- langchain-chroma
- streamlit

For a complete list, see `pyproject.toml`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
