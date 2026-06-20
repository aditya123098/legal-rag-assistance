"""
ingest.py
----------
Builds a FAISS vector store from the IPC/BNS legal dataset.

Each legal provision is converted into a LangChain Document with rich
metadata (IPC section, BNS section, category, punishment, etc.) so that
retrieved chunks carry structured fields the app can display directly,
not just raw text.

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
    Convert each legal section into a Document. We embed a composite string
    (title + category + text) so semantic search captures both the legal
    concept and the plain-language description, while keeping the full
    structured record in metadata for display in the UI.
    """
    docs = []
    for sec in sections:
        page_content = (
            f"{sec['title']}. Category: {sec['category']} - {sec['subcategory']}. "
            f"{sec['text']}"
        )
        metadata = {
            "id": sec["id"],
            "ipc_section": sec["ipc_section"],
            "bns_section": sec["bns_section"],
            "category": sec["category"],
            "subcategory": sec["subcategory"],
            "title": sec["title"],
            "punishment": sec["punishment"],
            "cognizable": sec["cognizable"],
            "bailable": sec["bailable"],
            "triable_by": sec["triable_by"],
        }
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Sections here are short enough that most won't actually split, but we
    keep a splitter in the pipeline so the system scales cleanly once you
    swap in the full IPC/BNS corpus (which will have much longer sections).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)


def main():
    print(f"Loading legal sections from {DATA_PATH} ...")
    sections = load_sections(DATA_PATH)
    print(f"Loaded {len(sections)} sections.")

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
