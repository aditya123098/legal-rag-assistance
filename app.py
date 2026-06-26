"""
app.py
-------
Streamlit UI for the Indian Cybercrime Legal RAG Assistant.

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
    page_title="Indian Cybercrime Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚖️ Cybercrime Legal Assistant")
    st.caption(
        "RAG over Indian Cyber Law — IT Act, IPC & BNS"
    )

    st.markdown("---")

    # Read API key from: sidebar input > Streamlit secrets > env var
    default_key = os.environ.get("GROQ_API_KEY", "")
    if not default_key:
        try:
            default_key = st.secrets.get("GROQ_API_KEY", "")
        except FileNotFoundError:
            default_key = ""

    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        value=default_key,
        help="Get a free key at console.groq.com. Used only for this session.",
    )
    top_k = st.slider("Entries to retrieve (k)", min_value=1, max_value=8, value=4)

    st.markdown("---")
    st.markdown(
        "**Stack:** LangChain · FAISS · Groq (LLaMA-3.3-70B) · Streamlit\n\n"
        "**Data:** 153 cybercrime situation entries covering IT Act, IPC & BNS sections."
    )
    st.markdown("---")
    st.caption(
        "⚠️ Educational/research tool only. Not a substitute for advice "
        "from a licensed advocate."
    )

# ---------- Main ----------
st.title("🔒 Indian Cybercrime Legal Research Assistant")
st.markdown(
    "Describe a situation or ask a question about Indian cyber law in plain English. "
    "The system retrieves the most relevant legal provisions and cites the applicable "
    "**IT Act, IPC (legacy), and BNS sections**."
)

example_qs = [
    "Someone hacked my Instagram account",
    "Is sharing someone's private photos a crime?",
    "I received a phishing email from my bank",
    "What are the laws against cyberbullying?",
    "Is using someone's WiFi without permission illegal?",
]

cols = st.columns(len(example_qs))
clicked_example = None
for col, q in zip(cols, example_qs):
    if col.button(q, use_container_width=True):
        clicked_example = q

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Describe a situation or ask about cyber law...")
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
                with st.spinner("Retrieving relevant entries and generating answer..."):
                    chain, retriever = build_rag_chain(
                        groq_api_key=api_key_input, k=top_k
                    )
                    retrieved_docs = retriever.invoke(final_question)
                    answer = chain.invoke(final_question)

                st.markdown(answer)

                with st.expander(f"📚 Retrieved entries ({len(retrieved_docs)})"):
                    for d in retrieved_docs:
                        m = d.metadata
                        is_crime = "✅ Yes" if m.get("is_cybercrime", False) else "❌ No"

                        st.markdown(
                            f"**Situation:** {m.get('situation', 'N/A')}  \n"
                            f"**Category:** {m.get('category', 'N/A')}  \n"
                            f"**Is Cybercrime:** {is_crime}  \n"
                        )

                        it_act = m.get("it_act_sections", "")
                        ipc = m.get("ipc_sections", "")
                        bns = m.get("bns_sections", "")

                        if it_act:
                            st.markdown(f"**IT Act:** {it_act}")
                        if ipc:
                            st.markdown(f"**IPC (legacy):** {ipc}")
                        if bns:
                            st.markdown(f"**BNS:** {bns}")

                        st.markdown(f"**Punishment:** {m.get('punishment', 'N/A')}")
                        st.markdown("---")

                st.session_state.history.append(
                    {"role": "assistant", "content": answer}
                )
            except FileNotFoundError:
                st.error(
                    "No FAISS index found. Run `python src/ingest.py` first to "
                    "build the vector store from the dataset."
                )
            except Exception as e:
                st.error(f"Something went wrong: {e}")
