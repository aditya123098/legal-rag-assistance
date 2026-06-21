"""
rag_chain.py
-------------
Core RAG logic: load the FAISS index, retrieve relevant legal entries
for a query, and generate a grounded answer using a Groq-hosted LLaMA
model.
"""

import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "faiss_index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You are a legal research assistant specialized in Indian cyber law and criminal law, \
covering the Information Technology Act, 2000 (IT Act), the Indian Penal Code (IPC), \
and the Bharatiya Nyaya Sanhita (BNS), 2023, which replaced the IPC.

Use ONLY the context provided below to answer the question. For every provision you \
reference, cite the applicable IT Act sections, IPC sections (legacy), and BNS sections \
where available.

Rules:
- Analyze the user's situation and determine whether it constitutes a cybercrime or not.
- If it is a cybercrime, clearly state the applicable legal sections from the IT Act, IPC, \
and BNS, along with explanations and punishments.
- If it is NOT a cybercrime, explain why it doesn't qualify as one.
- Be precise about punishments, applicable sections, and legal reasoning based on the context.
- If the context does not contain enough information to answer confidently, say so clearly \
instead of guessing.
- This tool is for legal research and educational purposes only. Always end substantive \
answers with a brief reminder to consult a qualified advocate for specific legal advice or \
case strategy, since statutory text alone does not capture case law, amendments, or local \
procedural nuance.

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def load_vectorstore() -> FAISS:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No FAISS index found at {INDEX_PATH}. Run `python src/ingest.py` first "
            f"to build the vector store from the sample dataset."
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(
        str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
    )


def format_docs(docs) -> str:
    """Render retrieved documents (with metadata) into a single context block."""
    blocks = []
    for d in docs:
        m = d.metadata
        is_crime = "Yes" if m.get("is_cybercrime", False) else "No"
        blocks.append(
            f"[Situation: {m.get('situation', 'N/A')}]\n"
            f"Category: {m.get('category', 'N/A')}\n"
            f"Is Cybercrime: {is_crime}\n"
            f"IT Act Sections: {m.get('it_act_sections', 'None')}\n"
            f"IPC Sections: {m.get('ipc_sections', 'None')}\n"
            f"BNS Sections: {m.get('bns_sections', 'None')}\n"
            f"Explanation: {m.get('explanation', 'N/A')}\n"
            f"Punishment: {m.get('punishment', 'N/A')}\n"
        )
    return "\n---\n".join(blocks)


def build_rag_chain(groq_api_key: str | None = None, k: int = 4):
    """
    Returns (chain, retriever). The chain produces a streaming-capable
    string output; the retriever is exposed separately so the UI can show
    "Retrieved sections" alongside the generated answer.
    """
    api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not set. Export it as an environment variable or pass "
            "it explicitly (e.g. via Streamlit secrets)."
        )

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    llm = ChatGroq(
        model=GROQ_MODEL_NAME,
        groq_api_key=api_key,
        temperature=0.1,  # low temperature: legal answers should be conservative
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def query(question: str, groq_api_key: str | None = None, k: int = 4):
    """Convenience function: run a single query end-to-end and return both
    the generated answer and the raw retrieved documents."""
    chain, retriever = build_rag_chain(groq_api_key=groq_api_key, k=k)
    retrieved_docs = retriever.invoke(question)
    answer = chain.invoke(question)
    return answer, retrieved_docs


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Is hacking someone's Instagram account a cybercrime?"
    answer, docs = query(q)
    print(f"\nQ: {q}\n")
    print(f"A: {answer}\n")
    print("Retrieved entries:")
    for d in docs:
        print(f" - [{d.metadata.get('category', 'N/A')}] {d.metadata.get('situation', 'N/A')[:80]}...")
