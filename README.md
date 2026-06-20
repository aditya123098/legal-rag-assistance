# ⚖️ IPC/BNS Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) system for querying Indian criminal
law — covering both the **Indian Penal Code (IPC)** and the **Bharatiya
Nyaya Sanhita (BNS), 2023**, which replaced the IPC. Built in the same
stack as [PdfTalker](#): LangChain + FAISS + Streamlit, powered by a
Groq-hosted LLaMA model.

The standout feature: every answer cross-references **both the old IPC
section and the new BNS section**, addressing a real pain point for
students, paralegals, and citizens navigating India's 2024 criminal law
overhaul.

## Features

- 🔍 Semantic search over structured legal provisions (not just raw text dumps)
- ⚖️ IPC ⇄ BNS cross-referencing for every retrieved section
- 📋 Rich metadata per provision: punishment, cognizability, bailability, trial court
- 💬 Chat-style interface with example questions and source transparency
- 🧩 Clean separation of ingestion (`ingest.py`) and inference (`rag_chain.py`) for easy extension

## Project Structure

```
legal-rag/
├── app.py                      # Streamlit UI
├── requirements.txt
├── data/
│   └── ipc_bns_sections.json   # Sample/illustrative legal dataset (20 sections)
├── src/
│   ├── ingest.py                # Builds the FAISS index from the dataset
│   └── rag_chain.py             # Retrieval + Groq LLM generation chain
└── .streamlit/
    └── config.toml
```

## Setup

1. **Clone and install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Get a free Groq API key** at [console.groq.com](https://console.groq.com)

3. **Build the vector index** (run once, or whenever the dataset changes)

   ```bash
   python src/ingest.py
   ```

4. **Run the app**

   ```bash
   streamlit run app.py
   ```

   Enter your Groq API key in the sidebar, or set it as an environment
   variable before launching:

   ```bash
   export GROQ_API_KEY="your-key-here"
   streamlit run app.py
   ```

## Using your own data

The included dataset (`data/ipc_bns_sections.json`) has **20 illustrative
sections** covering major crime categories (offences against the body,
property, women, public tranquility, documents, and cyber fraud) — enough
to demo the system end-to-end, but not the full statutory corpus.

To scale to the real IPC/BNS text:

1. Source the full text (e.g. from India Code, official gazette PDFs, or a vetted legal dataset).
2. Reshape it into the same JSON schema used in `ipc_bns_sections.json` — each entry needs `ipc_section`, `bns_section`, `category`, `title`, `text`, `punishment`, `cognizable`, `bailable`, and `triable_by`.
3. Re-run `python src/ingest.py` to rebuild the FAISS index.

The `ingest.py` pipeline already chunks long text via
`RecursiveCharacterTextSplitter`, so it will scale cleanly to full-length
statutory sections without code changes.

## Disclaimer

This tool is for **legal research and educational purposes only**. It does
not constitute legal advice, does not account for case law or recent
amendments beyond the indexed dataset, and should not be relied upon for
actual legal proceedings. Always consult a qualified advocate.

## Tech Stack

- **LangChain** — RAG orchestration
- **FAISS** — vector similarity search
- **sentence-transformers (all-MiniLM-L6-v2)** — local embeddings (free, no API key needed for ingestion)
- **Groq API (LLaMA-3.3-70B)** — fast, free-tier-friendly LLM inference
- **Streamlit** — UI
