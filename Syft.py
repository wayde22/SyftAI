import os
import json
import openai
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# --- Load Config ---
CONFIG_FILE = "config.json"
if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(f"Config file {CONFIG_FILE} not found!")
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

OPENAI_API_KEY = config["openai_api_key"]
CHROMA_PATH = config.get("chroma_path", "./chroma_db")
DEFAULT_CHUNKS = config.get("default_chunks", 5)
DEFAULT_MEMORY_WINDOW = config.get("memory_window", 5)

# Set the env var for OpenAI
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

# --- Memory Management ---
MEMORY_FILE = "memory.json"
memory_on = True
memory_window = DEFAULT_MEMORY_WINDOW
chat_history = []

def save_memory():
    with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
        json.dump({
            'memory_on': memory_on,
            'memory_window': memory_window,
            'chat_history': chat_history
        }, f)

def load_memory():
    global memory_on, memory_window, chat_history
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            memory_on = data.get('memory_on', True)
            memory_window = data.get('memory_window', DEFAULT_MEMORY_WINDOW)
            chat_history = data.get('chat_history', [])
    else:
        memory_on = True
        memory_window = DEFAULT_MEMORY_WINDOW
        chat_history = []

def get_vectorstore():
    embeddings = OpenAIEmbeddings()
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def get_code_context(question, n_chunks=DEFAULT_CHUNKS):
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=n_chunks)
    context = ""
    for doc in docs:
        context += f"\n[Source: {doc.metadata.get('source', 'unknown')}]:\n{doc.page_content}\n"
    return context or "(No relevant code found.)"

def get_llm_answer(prompt):
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful code assistant. If you don't know, say so."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=800,
        temperature=0.2
    )
    return response.choices[0].message.content

def build_prompt(question, context):
    prompt = ""
    if memory_on and chat_history:
        prompt += "Recent Q&A history:\n"
        for turn in chat_history[-memory_window:]:
            prompt += f"Q: {turn['q']}\nA: {turn['a']}\n"
    prompt += f"\nProject code context:{context}\n\nNew Question: {question}\nAnswer as helpfully as possible."
    return prompt

def main():
    global memory_on, memory_window, chat_history
    load_memory()
    print(f"Memory assistant started! (memory_on={memory_on}, memory_window={memory_window})")
    print("Commands: /memory on | off | clear | status | [number], /exit\n")

    while True:
        user_input = input("> ").strip()
        if user_input.lower().startswith("/memory"):
            cmd = user_input.split()
            if len(cmd) == 2 and cmd[1] == "on":
                memory_on = True
                print("Memory ENABLED.")
            elif len(cmd) == 2 and cmd[1] == "off":
                memory_on = False
                print("Memory DISABLED.")
            elif len(cmd) == 2 and cmd[1] == "clear":
                chat_history.clear()
                print("Memory CLEARED.")
            elif len(cmd) == 2 and cmd[1] == "status":
                print(f"Memory is {'ON' if memory_on else 'OFF'}, window = {memory_window}, history length = {len(chat_history)}")
            elif len(cmd) == 2 and cmd[1].isdigit():
                memory_window = int(cmd[1])
                print(f"Memory window set to last {memory_window} Q&As.")
            else:
                print("Usage: /memory on | off | clear | status | [number]")
            save_memory()
            continue
        elif user_input.lower() in ("/exit", "/quit"):
            print("Exiting. Bye!")
            save_memory()
            break
        elif not user_input:
            continue

        # --- Code context retrieval ---
        print("Searching codebase for context...")
        context = get_code_context(user_input)

        # --- Prompt build & LLM call ---
        prompt = build_prompt(user_input, context)
        print("Contacting GPT... (this may take a moment)")
        answer = get_llm_answer(prompt)
        print(f"\nAssistant: {answer}\n")

        # --- Update memory and persist ---
        chat_history.append({'q': user_input, 'a': answer})
        save_memory()

if __name__ == "__main__":
    main()
