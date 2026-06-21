<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-0.2+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/FAISS-Vector_DB-0467DF?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

<h1 align="center">⚖️ Indian Cybercrime Legal RAG Assistant</h1>

<p align="center">
  <strong>An AI-powered legal research tool for Indian cyber law — built with Retrieval-Augmented Generation (RAG)</strong>
</p>

<p align="center">
  Describe any situation in plain English and instantly find out whether it's a cybercrime,<br/>
  which laws apply, and what the punishments are — all backed by the<br/>
  <strong>IT Act 2000</strong> · <strong>Indian Penal Code (IPC)</strong> · <strong>Bharatiya Nyaya Sanhita (BNS) 2023</strong>
</p>

---

## 🎯 What It Does

> **"Someone hacked my Instagram and is posting from my account — is this a crime?"**

The assistant will:
1. 🔍 **Search** through 153 real-world cybercrime scenarios using semantic similarity
2. 📋 **Retrieve** the most relevant legal provisions (IT Act, IPC & BNS sections)
3. 🤖 **Generate** a detailed, grounded legal analysis using LLaMA-3.3-70B
4. ✅ **Classify** whether the situation is a cybercrime or not
5. ⚖️ **Cite** applicable sections with explanations and punishments

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Semantic Search** | FAISS-powered vector search over 153 cybercrime situations |
| ⚖️ **Triple Law Coverage** | Cross-references IT Act + IPC (legacy) + BNS 2023 for every query |
| 🏷️ **Cybercrime Classifier** | Clearly tells you if a situation is a cybercrime or not |
| 📚 **Source Transparency** | Shows all retrieved entries with full metadata |
| 💬 **Chat Interface** | Conversational UI with history and example questions |
| 🔒 **Privacy-First** | Embeddings run locally — no data leaves your machine for indexing |
| ⚡ **Fast Inference** | Powered by Groq's ultra-fast LLaMA-3.3-70B API |
| 🆓 **Free to Run** | Uses free-tier Groq API + free local embeddings |

---

## 📂 Project Structure

```
legal-rag/
├── 📄 app.py                       # Streamlit UI — the main application
├── 📄 requirements.txt             # Python dependencies
├── 📁 data/
│   ├── 📄 ipc_bns_sections.json    # 153 cybercrime situation entries (dataset)
│   └── 📁 faiss_index/             # FAISS vector index (auto-generated)
│       ├── index.faiss
│       └── index.pkl
├── 📁 src/
│   ├── 📄 ingest.py                # Builds FAISS index from the dataset
│   └── 📄 rag_chain.py             # RAG retrieval + LLM generation chain
├── 📁 .streamlit/
│   └── 📄 config.toml              # Streamlit theme configuration
└── 📄 README.md
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Groq API Key** (free) — get one at [console.groq.com](https://console.groq.com)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/aditya123098/legal-rag-assistance.git
cd legal-rag-assistance
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Build the Vector Index

> Run this once, or whenever the dataset changes.

```bash
python src/ingest.py
```

This will:
- Load all 153 entries from `data/ipc_bns_sections.json`
- Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2` (runs locally)
- Save the FAISS index to `data/faiss_index/`

### 4️⃣ Run the App

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser and enter your Groq API key in the sidebar.

**Or** set the API key as an environment variable:

```bash
# Linux/Mac
export GROQ_API_KEY="your-key-here"

# Windows (PowerShell)
$env:GROQ_API_KEY="your-key-here"

streamlit run app.py
```

---

## 📊 Dataset

The dataset contains **153 entries** covering real-world cybercrime scenarios across **12 categories**:

| Category | Examples |
|----------|----------|
| 🔓 Hacking / Unauthorized Access | Server breaches, account takeovers, keyloggers |
| 💰 Financial Fraud / Online Banking Fraud | Phishing, UPI fraud, crypto scams, BEC attacks |
| 😡 Cyberbullying / Harassment / Stalking | Online threats, fake profiles, cyberstalking |
| 🔐 Data Theft / Privacy Violation | Data breaches, sextortion, unauthorized data sharing |
| 🔞 Obscene/Sexual Content & Child Safety | CSAM, unsolicited explicit content |
| 📰 Defamation / Fake News / Impersonation | Online defamation, misinformation, identity fraud |
| 🆔 Identity Theft / Phishing / Spoofing | Aadhaar fraud, caller ID spoofing, credential theft |
| 🦠 Malware / Ransomware / Virus | Ransomware attacks, virus distribution |
| 🛒 E-commerce / Online Shopping Fraud | Fake websites, non-delivery scams |
| 💣 Cyber Terrorism / Critical Infrastructure | Attacks on critical infrastructure |
| 🏢 Workplace / IP / Trade Secrets | Employee data theft, trade secret violations |
| ❌ Not a Cybercrime | Legitimate activities (negative examples for accuracy) |

### Dataset Schema

Each entry follows this structure:

```json
{
  "id": 1,
  "situation": "Someone gained unauthorized access to my company's server...",
  "is_cybercrime": true,
  "category": "Hacking / Unauthorized Access",
  "it_act_sections": ["Section 43 (...)", "Section 66 (...)"],
  "ipc_sections": ["Section 379 (theft) - legacy reference"],
  "bns_sections": ["Section 303 BNS (theft)"],
  "explanation": "Unauthorized access to a computer system...",
  "punishment": "Under Section 66 IT Act: imprisonment up to 3 years..."
}
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Orchestration** | [LangChain](https://www.langchain.com/) | RAG pipeline, prompt management, chain composition |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) | Fast similarity search over embedded legal entries |
| **Embeddings** | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Free, local embedding model (no API key needed) |
| **LLM** | [Groq](https://groq.com/) (LLaMA-3.3-70B) | Ultra-fast, free-tier LLM inference |
| **UI** | [Streamlit](https://streamlit.io/) | Interactive web interface |

---

## 🔄 How It Works

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  User Query  │────▶│  FAISS Retriever  │────▶│  Top-K Matching  │
│  (plain      │     │  (semantic search │     │  Entries with    │
│   English)   │     │   over 153 entries│     │  full metadata)  │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                            ┌─────────────────┐
                                            │  LLaMA-3.3-70B  │
                                            │  (via Groq API)  │
                                            │                  │
                                            │  Generates legal │
                                            │  analysis using  │
                                            │  retrieved context│
                                            └────────┬────────┘
                                                     │
                                                     ▼
                                            ┌─────────────────┐
                                            │  Structured      │
                                            │  Legal Answer    │
                                            │  + Source Docs   │
                                            └─────────────────┘
```

---

## 💡 Example Queries

| Query | What You'll Get |
|-------|-----------------|
| *"Someone hacked my Instagram account"* | IT Act Sec 43, 66, 66C — up to 3 years imprisonment |
| *"I received a phishing email from my bank"* | Sec 66C, 66D — identity theft + cheating by personation |
| *"Is using someone's WiFi without permission illegal?"* | Sec 43(a) — technically unauthorized access |
| *"My ex is threatening to share my private photos"* | Sec 66E, 67A + BNS 308 — sextortion, up to 7 years |
| *"I paid my friend on Google Pay for dinner"* | ❌ Not a cybercrime — legitimate transaction |

---

## ⚠️ Disclaimer

> This tool is for **legal research and educational purposes only**. It does not constitute legal advice, does not account for case law or recent amendments beyond the indexed dataset, and should not be relied upon for actual legal proceedings. **Always consult a qualified advocate** for specific legal advice or case strategy.

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Expand the dataset** — Add more cybercrime scenarios to `data/ipc_bns_sections.json`
2. **Improve the UI** — Enhance the Streamlit interface
3. **Add new features** — Multi-language support, PDF export, case law references

```bash
# Fork the repo, make changes, then:
git add -A
git commit -m "Add: description of changes"
git push origin your-branch
# Open a Pull Request on GitHub
```

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ for legal awareness in India
</p>
