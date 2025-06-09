import os
import json
import time
import openai
import shutil
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Load Config ---
CONFIG_FILE = "config.json"
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Config file {CONFIG_FILE} not found!")
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

OPENAI_API_KEY = config["openai_api_key"]
PROJECT_PATH = config["codebase_path"]
CHROMA_PATH = config.get("chroma_path", "./chroma_db")
DEFAULT_CHUNKS = config.get("default_chunks", 5)

# --- Always start with a clean Chroma DB ---
if os.path.exists(CHROMA_PATH):
    print(f"Removing existing Chroma DB at {CHROMA_PATH} ...")
    shutil.rmtree(CHROMA_PATH)

# Set the environment variable so LangChain/OpenAI picks up your API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Load all relevant code files ---
def load_code_files(project_path):
    code_files = []
    EXCLUDE_DIRS = {'.venv', 'env', 'venv', '__pycache__', 'site-packages'}
    for root, dirs, files in os.walk(project_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for file in files:
            if file.endswith(('.py', '.js', '.ts', '.java', '.kt', '.c', '.cpp', '.json', '.yml', '.yaml')):
                file_path = os.path.join(root, file)
                print("Indexing file:", file_path)   # <--- ADD THIS LINE
                try:
                    with open(file_path, encoding='utf-8', errors='ignore') as f:
                        code_files.append({'path': file_path, 'content': f.read()})
                except Exception as e:
                    print(f"Warning: Could not read {file_path}: {e}")
    return code_files

# --- Split code into chunks for better embeddings retrieval ---
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = []
for file in load_code_files(PROJECT_PATH):
    chunks = splitter.split_text(file['content'])
    for chunk in chunks:
        docs.append({'content': chunk, 'metadata': {'source': file['path']}})

if not docs:
    print("No code files found to index! Check your codebase_path in config.json.")
    exit(1)

# --- Adaptive batcher for OpenAI token errors ---
def adaptive_add_texts(vectorstore, texts, metadatas, min_batch=16):
    """
    Add as many texts as possible in one batch. If batch is too big (OpenAI token error), splits and retries recursively.
    """
    batch_size = len(texts)
    while True:
        try:
            if vectorstore is None:
                result = Chroma.from_texts(
                    texts, embeddings, metadatas=metadatas, persist_directory=CHROMA_PATH
                )
            else:
                vectorstore.add_texts(texts, metadatas=metadatas)
                result = vectorstore
            return result
        except openai.BadRequestError as e:
            if batch_size <= min_batch:
                print(f"[ERROR] Batch failed even at minimum size: {e}")
                raise
            batch_size //= 2
            print(f"[WARN] Batch too large, retrying with size {batch_size}...")
            mid = len(texts) // 2
            adaptive_add_texts(vectorstore, texts[:mid], metadatas[:mid], min_batch)
            adaptive_add_texts(vectorstore, texts[mid:], metadatas[mid:], min_batch)
            return vectorstore
        except Exception as e:
            print(f"[ERROR] Unexpected error: {e}")
            time.sleep(1)
            raise

# --- Create the vectorstore in Chroma ---
print(f"Indexing {len(docs)} code chunks...")
embeddings = OpenAIEmbeddings()
BATCH_SIZE = 1000  # Try big, adaptive will split as needed

all_texts = [d['content'] for d in docs]
all_metas = [d['metadata'] for d in docs]
vectorstore = None

i = 0
n = len(all_texts)
while i < n:
    batch_texts = all_texts[i:i+BATCH_SIZE]
    batch_metas = all_metas[i:i+BATCH_SIZE]
    print(f"Indexing {i+1}-{i+len(batch_texts)} / {n} ...")
    vectorstore = adaptive_add_texts(vectorstore, batch_texts, batch_metas)
    i += BATCH_SIZE

print(f"Indexing complete! Chroma DB stored at: {CHROMA_PATH}")
