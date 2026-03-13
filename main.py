import streamlit as st
import os
import glob
from pathlib import Path
from src.agent.agent import OmniAgent
from PIL import Image


st.set_page_config(page_title="Omni-Analyst Dashboard", layout="wide", page_icon="🔬")

if "agent" not in st.session_state:
    with st.spinner("🧠 Initializing Omni-Analyst Brain..."):
        st.session_state.agent = OmniAgent(db_path="data/vector_db")

if "messages" not in st.session_state:
    st.session_state.messages = []


def get_pdf_count():
    return len(glob.glob("data/raw/*.pdf"))

with st.sidebar:
    st.title("📂 Document Library")
    
    pdf_count = get_pdf_count()
    st.success(f"Currently analyzing {pdf_count} PDFs in `data/raw`.")
    
    with st.expander("View Indexed Files"):
        for pdf in glob.glob("data/raw/*.pdf"):
            st.write(f"📄 {os.path.basename(pdf)}")

    st.divider()
    st.subheader("Developer Tools")
    show_thought_process = st.checkbox("Show Query Rewriting", value=True)
    show_context = st.checkbox("Show Retrieval Context", value=False)
    
    if st.button("Clear Chat & Memory"):
        st.session_state.messages = []
        st.session_state.agent.history = [] 
        st.rerun()

# --- MAIN CHAT INTERFACE ---
st.title("🔬 Omni-Analyst")
st.caption("Multimodal RAG System | Powered by Gemini 2.5 Flash")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if "rewritten" in message and show_thought_process:
            st.caption(f"🔍 *Search Query: {message['rewritten']}*")
       
        if "image_path" in message and message["image_path"]:
            st.image(message["image_path"], caption="Referenced Image", width=400)


if prompt := st.chat_input("Ask a question about your documents..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            full_answer, rewritten_query = st.session_state.agent.ask(prompt)
            
            if show_thought_process:
                st.caption(f"🔄 **Rewritten Query:** {rewritten_query}")
            
            st.markdown(full_answer)
            
            raw_results = st.session_state.agent.vs.query(rewritten_query, n_results=3)
            
            image_to_show = None
            for meta in raw_results['metadatas'][0]:
                if meta.get('type') == 'image_caption':
                    img_path = meta.get('image_path')
                    if img_path and os.path.exists(img_path):
                        image_to_show = img_path
                        st.image(image_to_show, caption="Relevant Visual Source", width=500)
                        break

            if show_context:
                with st.expander("🔍 Evaluation: See Retrieved Context Chunks"):
                    for i, (text, meta) in enumerate(zip(raw_results['documents'][0], raw_results['metadatas'][0])):
                        st.write(f"**Chunk {i+1}** (Source: {meta.get('source_file')}, Pg: {meta.get('page')})")
                        st.code(text[:500] + "...")


    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_answer,
        "rewritten": rewritten_query,
        "image_path": image_to_show
    })