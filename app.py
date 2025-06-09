import os
import json
import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import openai

# --- Load config ---
CONFIG_FILE = "config.json"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

OPENAI_API_KEY = config["openai_api_key"]
CHROMA_PATH = config.get("chroma_path", "./chroma_db")
DEFAULT_CHUNKS = config.get("default_chunks", 5)
DEFAULT_MEMORY_WINDOW = config.get("memory_window", 5)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- Streamlit page setup ---
st.set_page_config(page_title="Syft GPT Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Syft GPT Assistant")

# --- Custom CSS for dark mode-friendly chat bubbles ---
st.markdown("""
<style>
/* Chat assistant message */
.chat-bubble-assistant {
    background: #222b35;
    color: #fff;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 18px;
    border: 1px solid #181f26;
    font-size: 1.05em;
}
/* Code blocks inside chat bubbles */
.chat-bubble-assistant pre,
.chat-bubble-assistant code {
    background: #11151a !important;
    color: #ffe88c !important;
    border-radius: 8px;
    padding: 10px;
    font-size: 1em;
}
</style>
""", unsafe_allow_html=True)


# --- Session state (acts as memory between interactions) ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory_on" not in st.session_state:
    st.session_state.memory_on = False
if "memory_window" not in st.session_state:
    st.session_state.memory_window = DEFAULT_MEMORY_WINDOW

# --- Helper Functions ---
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
    try:
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
        answer = response.choices[0].message.content
        return answer
    except Exception as e:
        print(f"\n[Error contacting OpenAI]: {e}\n")
        return "(Error: Unable to contact OpenAI or process the prompt.)"

def build_prompt(question, context, history, memory_on, memory_window):
    prompt = ""
    if memory_on and history:
        prompt += "Recent Q&A history:\n"
        for turn in history[-memory_window:]:
            prompt += f"Q: {turn['q']}\nA: {turn['a']}\n"
    prompt += f"\nProject code context:{context}\n\nNew Question: {question}\nAnswer as helpfully as possible."
    return prompt

def clear_history():
    st.session_state.chat_history = []

def clear_input():
    st.session_state.input = ""

# --- Sidebar Memory Controls ---
st.sidebar.header("Memory Settings")
memory_toggle = st.sidebar.toggle("Memory ON/OFF", value=st.session_state.memory_on)
memory_window = st.sidebar.slider("Memory Window (last N Q&As)", 1, 20, st.session_state.memory_window)
if st.sidebar.button("Clear Memory"):
    clear_history()
    st.sidebar.success("Memory cleared.")

# Update session state if controls changed
st.session_state.memory_on = memory_toggle
st.session_state.memory_window = memory_window

# --- Chat Display (left column) ---
chat_col, prompt_col = st.columns([2, 1])

with chat_col:
    for i, turn in enumerate(st.session_state.chat_history):
        st.markdown(f"**You:** {turn['q']}")
        st.markdown(
            f"<div class='chat-bubble-assistant'>{turn['a']}</div>",
            unsafe_allow_html=True
        )

# --- Prompt Input (right column) ---
with prompt_col:
    st.subheader("Ask Your Codebase")
    user_input = st.text_area(
        "Type your question and press Enter (Shift+Enter for newline):",
        key="input",
        height=100
    )
    ask_btn = st.button("Ask")
    clear_btn = st.button("Clear input", on_click=clear_input)

    if ask_btn and user_input.strip():
        with st.spinner("Thinking... searching your codebase and contacting GPT..."):
            context = get_code_context(user_input)
            prompt = build_prompt(
                user_input,
                context,
                st.session_state.chat_history,
                st.session_state.memory_on,
                st.session_state.memory_window
            )
            answer = get_llm_answer(prompt)
        # Add the Q&A to chat history
        st.session_state.chat_history.append({'q': user_input, 'a': answer})
        st.rerun()

# --- Footer/Status ---
st.markdown(f"""
---
<small>Powered by OpenAI + LangChain + Streamlit. Your codebase: <b>{config['codebase_path']}</b></small>
""", unsafe_allow_html=True)
