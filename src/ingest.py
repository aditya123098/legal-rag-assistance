"""
ingest.py
----------
Builds a FAISS vector store from the Indian cybercrime legal dataset.

Each entry is converted into a LangChain Document with rich metadata
(IT Act sections, IPC sections, BNS sections, category, punishment, etc.)
so that retrieved chunks carry structured fields the app can display
directly, not just raw text.

Run this once (or whenever data/ipc_bns_sections.json changes):
    python src/ingest.py
"""

import json
import os
from pathlib import Path

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ipc_bns_sections.json"
INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "faiss_index"

# Small, fast, free local embedding model — no API key required for ingestion.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_sections(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_documents(sections: list[dict]) -> list[Document]:
    """
    Convert each legal entry into a Document. We embed a composite string
    (situation + category + explanation + sections) so semantic search
    captures the legal concept, applicable laws, and the plain-language
    description, while keeping the full structured record in metadata
    for display in the UI.
    """
    docs = []
    for sec in sections:
        # Build a rich text representation for embedding
        it_act_str = "; ".join(sec.get("it_act_sections", []))
        ipc_str = "; ".join(sec.get("ipc_sections", []))
        bns_str = "; ".join(sec.get("bns_sections", []))

        page_content = (
            f"Situation: {sec['situation']}\n"
            f"Category: {sec['category']}\n"
            f"Is Cybercrime: {'Yes' if sec.get('is_cybercrime', False) else 'No'}\n"
            f"IT Act Sections: {it_act_str or 'None'}\n"
            f"IPC Sections: {ipc_str or 'None'}\n"
            f"BNS Sections: {bns_str or 'None'}\n"
            f"Explanation: {sec['explanation']}\n"
            f"Punishment: {sec['punishment']}"
        )

        metadata = {
            "id": sec["id"],
            "situation": sec["situation"],
            "is_cybercrime": sec.get("is_cybercrime", False),
            "category": sec["category"],
            "it_act_sections": it_act_str,
            "ipc_sections": ipc_str,
            "bns_sections": bns_str,
            "explanation": sec["explanation"],
            "punishment": sec["punishment"],
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Entries here are short enough that most won't actually split, but we
    keep a splitter in the pipeline so the system scales cleanly if you
    add longer entries in the future.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def main():
    print(f"Loading legal sections from {DATA_PATH} ...")
    sections = load_sections(DATA_PATH)
    print(f"Loaded {len(sections)} entries.")

    docs = build_documents(sections)
    chunks = chunk_documents(docs)
    print(f"Prepared {len(chunks)} chunks for embedding.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print("Building FAISS index ...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(INDEX_PATH, exist_ok=True)
    vectorstore.save_local(str(INDEX_PATH))
    print(f"FAISS index saved to {INDEX_PATH}")


if __name__ == "__main__":
    main()
