# 🧠 ResearchMind — AI Research Assistant

> An intelligent research chatbot that lets you upload academic papers and get cited, accurate answers powered by RAG and live web search.

🔗 **Live Demo**: [research-chatbot-toshima2612.streamlit.app](https://research-chatbot-toshima2612.streamlit.app)

---

## Features

- **Upload any PDF** — research papers, textbooks, reports, legal docs
- **RAG-powered answers** — retrieves the most relevant chunks from your documents
- **Live Web Search** — DuckDuckGo integration for real-time results
- **Concise mode** — 2-3 sentence sharp answers
- **Detailed mode** — in-depth paragraph explanations with citations
- **Source transparency** — see exactly which paper or URL answered your question
- **Chat history** — saved automatically, reload any past conversation

---

## 🛠 Tech Stack

| Component | Tool |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector DB | FAISS |
| PDF Parsing | PyMuPDF + pytesseract OCR |
| Web Search | DuckDuckGo (no API key needed) |
| Frontend | Streamlit |
| Chat History | SQLite |

---

## Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/toshimajaiswal/RESEARCHMIND.git
cd RESEARCHMIND
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_key_here
```
Get a free key at [console.groq.com/keys](https://console.groq.com/keys)

**5. Run the app**
```bash
streamlit run app.py
```

---

## Project Structure
```
RESEARCHMIND/
├── app.py                  # Main Streamlit application
├── requirements.txt
├── .gitignore
├── config/
│   └── config.py           # API keys and model settings
├── models/
│   ├── llm.py              # Groq LLM setup
│   └── embeddings.py       # BGE embedding model
└── utils/
    ├── rag_pipeline.py     # PDF parsing, FAISS indexing, retrieval
    ├── web_search.py       # DuckDuckGo search
    ├── prompt.py           # System prompt builder
    ├── chat_history.py     # SQLite chat history
    └── __init__.py
```

---

## API Keys Required

| Key | Source | Free? |
|---|---|---|
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) |  Yes |

---

## ⚙️ How It Works
```
User uploads PDF
      ↓
PyMuPDF extracts text → chunked → BGE embeddings → FAISS index
      ↓
User asks a question
      ↓
Question embedded → FAISS retrieves top-3 relevant chunks
      ↓
DuckDuckGo fetches live web results (if enabled)
      ↓
Groq LLM generates a grounded, cited answer
      ↓
Sources displayed to user
```

---

## Notes

- Works with **any PDF** in any domain — not limited to AI/research papers
- For best results, upload text-based PDFs (downloaded from arXiv, not scanned)
- Free Groq tier allows ~100,000 tokens/day — use concise mode for longer sessions
- Chat history resets if the app is restarted on Streamlit Cloud (SQLite is ephemeral)

---

## Author

**Toshima Jaiswal**
Built for the NeoStats AI Engineer Chatbot Blueprint Challenge
