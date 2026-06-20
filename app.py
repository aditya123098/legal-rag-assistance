"""
app.py
-------
Streamlit UI for the IPC/BNS Legal RAG Assistant.

Run with:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent / "src"))

from rag_chain import build_rag_chain, format_docs  # noqa: E402

st.set_page_config(
    page_title="IPC/BNS Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚖️ Legal RAG Assistant")
    st.caption("RAG over Indian Penal Code (IPC) & Bharatiya Nyaya Sanhita (BNS)")

    st.markdown("---")
    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key at console.groq.com. Used only for this session.",
    )
    top_k = st.slider("Sections to retrieve (k)", min_value=1, max_value=8, value=4)

    st.markdown("---")
    st.markdown(
        "**Stack:** LangChain · FAISS · Groq (LLaMA-3.3-70B) · Streamlit\n\n"
        "**Data:** Sample/illustrative IPC & BNS sections — swap in the "
        "full statutory corpus via `src/ingest.py` for production use."
    )
    st.markdown("---")
    st.caption(
        "⚠️ Educational/research tool only. Not a substitute for advice "
        "from a licensed advocate."
    )

# ---------- Main ----------
st.title("IPC ⇄ BNS Legal Research Assistant")
st.markdown(
    "Ask a question about Indian criminal law in plain English. The system "
    "retrieves the relevant statutory provisions and cites **both the old "
    "IPC section and the corresponding new BNS section**."
)

example_qs = [
    "What is the punishment for theft?",
    "What's the difference between robbery and dacoity?",
    "Which IPC section corresponds to BNS Section 103?",
    "Is criminal breach of trust a bailable offence?",
    "What are the punishments for dowry death?",
]

cols = st.columns(len(example_qs))
clicked_example = None
for col, q in zip(cols, example_qs):
    if col.button(q, use_container_width=True):
        clicked_example = q

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask about IPC/BNS sections, punishments, procedure...")
final_question = clicked_example or question

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

if final_question:
    st.session_state.history.append({"role": "user", "content": final_question})
    with st.chat_message("user"):
        st.markdown(final_question)

    with st.chat_message("assistant"):
        if not api_key_input:
            st.error("Please enter your Groq API key in the sidebar to continue.")
        else:
            try:
                with st.spinner("Retrieving relevant sections and generating answer..."):
                    chain, retriever = build_rag_chain(
                        groq_api_key=api_key_input, k=top_k
                    )
                    retrieved_docs = retriever.invoke(final_question)
                    answer = chain.invoke(final_question)

                st.markdown(answer)

                with st.expander(f"📚 Retrieved sections ({len(retrieved_docs)})"):
                    for d in retrieved_docs:
                        m = d.metadata
                        st.markdown(
                            f"**{m['title']}**  \n"
                            f"IPC: `{m['ipc_section']}` &nbsp;|&nbsp; "
                            f"BNS: `{m['bns_section']}`  \n"
                            f"Category: {m['category']} — {m['subcategory']}  \n"
                            f"Punishment: {m['punishment']}  \n"
                            f"Cognizable: {'Yes' if m['cognizable'] else 'No'} | "
                            f"Bailable: {'Yes' if m['bailable'] else 'No'} | "
                            f"Triable by: {m['triable_by']}"
                        )
                        st.markdown("---")

                st.session_state.history.append(
                    {"role": "assistant", "content": answer}
                )
            except FileNotFoundError:
                st.error(
                    "No FAISS index found. Run `python src/ingest.py` first to "
                    "build the vector store from the sample dataset."
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
