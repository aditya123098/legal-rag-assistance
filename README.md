# ⚖️ Indian Cybercrime Legal RAG Assistant

A Retrieval-Augmented Generation (RAG) system for querying Indian cyber law
— covering the **Information Technology Act, 2000**, the **Indian Penal Code
(IPC)**, and the **Bharatiya Nyaya Sanhita (BNS), 2023**, which replaced the
IPC. Built with LangChain + FAISS + Streamlit, powered by a Groq-hosted
LLaMA model.

The standout feature: describe any situation in plain English, and the system
tells you **whether it's a cybercrime**, which **IT Act / IPC / BNS sections**
apply, the **explanation**, and the **punishment**.

## Features

- 🔍 Semantic search over 153 cybercrime situation entries
- ⚖️ IT Act + IPC (legacy) + BNS cross-referencing for every retrieved entry
- 📋 Rich metadata: category, is_cybercrime flag, explanation, punishment
- 💬 Chat-style interface with example questions and source transparency
- 🧩 Clean separation of ingestion (`ingest.py`) and inference (`rag_chain.py`)

## Project Structure

```
legal-rag/
├── app.py                      # Streamlit UI
├── requirements.txt
├── data/
│   ├── ipc_bns_sections.json   # 153 cybercrime situation entries
│   └── faiss_index/            # Built by ingest.py (auto-generated)
├── src/
│   ├── ingest.py               # Builds the FAISS index from the dataset
│   └── rag_chain.py            # Retrieval + Groq LLM generation chain
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

## Dataset Schema

Each entry in `data/ipc_bns_sections.json` has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique identifier |
| `situation` | string | Plain-English description of a scenario |
| `is_cybercrime` | boolean | Whether the situation qualifies as a cybercrime |
| `category` | string | Category (e.g., "Hacking / Unauthorized Access") |
| `it_act_sections` | string[] | Applicable IT Act sections |
| `ipc_sections` | string[] | Applicable IPC sections (legacy) |
| `bns_sections` | string[] | Applicable BNS sections |
| `explanation` | string | Legal explanation of why the sections apply |
| `punishment` | string | Applicable punishments |

## Disclaimer

This tool is for **legal research and educational purposes only**. It does
not constitute legal advice, does not account for case law or recent
amendments beyond the indexed dataset, and should not be relied upon for
actual legal proceedings. Always consult a qualified advocate.

## Tech Stack

- **LangChain** — RAG orchestration
- **FAISS** — vector similarity search
- **sentence-transformers (all-MiniLM-L6-v2)** — local embeddings (free, no API key needed)
- **Groq API (LLaMA-3.3-70B)** — fast, free-tier-friendly LLM inference
- **Streamlit** — UI
