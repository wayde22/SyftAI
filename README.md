# SyftAI

SyftAI is an interactive code assistant that leverages OpenAI, LangChain, and Qdrant (local vector DB) to answer questions about your codebase using context-aware retrieval-augmented generation.
It features memory (chat history), customizable context window, incremental indexing, .gitignore support, and a modern Streamlit UI.

---

## Getting Started

### 1. Set up Python Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
````
---

## Getting Started
### Indexing the Codebase

Before running the assistant, you need to **index your codebase**. This step scans your code and builds a searchable vector database for question answering.

To index your codebase, run:

```bash
python Indexer.py
```

### Running the Assistant App

After indexing, you can start the Syft AI Assistant web app with:

```bash
streamlit run app.py
```

### **Requirements**
- Python 3.9+
- OpenAI API key
- `pip install -r requirements.txt`

python -m venv .venv
source .venv/bin/activate         # On Windows: .venv\Scripts\activate

### The JSON object should go in your config.json
- Add your OpenAI api keys.
- The path to the project that you want it targeting.
- Path to Chroma_db.
- How many chunks do you want to index.
- And how many prompts and replies do you want it to be added to memory.

```json
{
  "openai_api_key": "sk-...",
  "codebase_path": "./path_to_your_code",
  "chroma_path": "./qdrant_data",
  "default_chunks": 5,
  "memory_window": 5
}
```

## Dependencies

To run Syft, make sure you have the following Python packages installed:

- `streamlit`
- `openai`
- `langchain`
- `langchain_chroma`
- `langchain_openai`
- `chromadb` (for Chroma vector storage)
- `tiktoken` (required by some OpenAI models)
- `pathspec` (for .gitignore support)

### Install All Dependencies

You can install these with pip:

```bash
# SyftAI

SyftAI is an interactive code assistant that leverages OpenAI, LangChain, and Qdrant (local vector DB) to answer questions about your codebase using context-aware retrieval-augmented generation.  
It features memory (chat history), customizable context window, incremental indexing, and a modern Streamlit UI.

---

## Getting Started

### 1. Set up Python Environment

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
````
---

## Getting Started
### Indexing the Codebase

Before running the assistant, you need to **index your codebase**. This step scans your code and builds a searchable vector database for question answering.

To index your codebase, run:

```bash
python Indexer.py
```

### Running the Assistant App

After indexing, you can start the Syft AI Assistant web app with:

```bash
streamlit run app.py
```

### **Requirements**
- Python 3.9+
- OpenAI API key
- `pip install -r requirements.txt`

python -m venv .venv
source .venv/bin/activate         # On Windows: .venv\Scripts\activate

### The JSON object should go in your config.json
- Add your OpenAI api keys.
- The path to the project that you want it targeting.
- Path to Chroma_db.
- How many chunks do you want to index.
- And how many prompts and replies do you want it to be added to memory.

```json
{
  "openai_api_key": "sk-...",
  "codebase_path": "./path_to_your_code",
  "chroma_path": "./qdrant_data",
  "default_chunks": 5,
  "memory_window": 5
}
```

## Dependencies

To run Syft, make sure you have the following Python packages installed:

- `streamlit`
- `openai`
- `langchain`
- `langchain_chroma`
- `langchain_openai`
- `chromadb` (for Chroma vector storage)
- `tiktoken` (required by some OpenAI models)

### Install All Dependencies

You can install these with pip:

```bash
pip install streamlit openai langchain langchain-qdrant langchain_openai qdrant-client tiktoken pathspec
```

```
