import streamlit as st
import os
from pathlib import Path
from src.agent.agent import OmniAgent
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="Omni-Analyst Dashboard", layout="wide", page_icon="🔬")

# --- INITIALIZATION ---
# Using session_state so the agent doesn't reload on every click
if "agent" not in st.session_state:
    with st.spinner("🧠 Initializing Omni-Analyst Brain..."):
        st.session_state.agent = OmniAgent(db_path="data/vector_db")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR ---
with st.sidebar:
    st.title("📂 Document Library")
    st.info("Currently analyzing 8 medical/architectural PDFs.")
    
    st.divider()
    st.subheader("Developer Tools")
    show_context = st.checkbox("Show Retrieval Context", value=False)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("🔬 Omni-Analyst")
st.caption("Multimodal RAG System for Technical Document Intelligence")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there's an image associated with this response, show it
        if "image_path" in message and message["image_path"]:
            st.image(message["image_path"], caption="Referenced Image", width=400)

# Chat Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            # 1. Get the Raw Query Results for Evaluation/UI
            # (Adding a small helper method to the agent to get raw results)
            raw_results = st.session_state.agent.vs.query(prompt, n_results=3)
            
            # 2. Get the LLM Answer
            full_answer = st.session_state.agent.ask(prompt)
            
            st.markdown(full_answer)
            
            # 3. Handle Multimodal References
            # If the top result is an image, let's display it!
            image_to_show = None
            for meta in raw_results['metadatas'][0]:
                if meta.get('type') == 'image_caption':
                    image_to_show = meta.get('image_path')
                    if image_to_show and os.path.exists(image_to_show):
                        st.image(image_to_show, caption="Source Image found by Agent", width=500)
                        break

            # 4. Developer Mode: Show Context
            if show_context:
                with st.expander("🔍 Evaluation: See Retrieved Context Chunks"):
                    for i, (text, meta) in enumerate(zip(raw_results['documents'][0], raw_results['metadatas'][0])):
                        st.write(f"**Chunk {i+1}** (Source: {meta.get('source_file')}, Page: {meta.get('page')})")
                        st.code(text[:500] + "...")

    # Save Assistant message
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_answer,
        "image_path": image_to_show if image_to_show else None
    })