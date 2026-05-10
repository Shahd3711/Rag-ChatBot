# NOVA — Nebula-Oriented Virtual Astronomer
### Agentic RAG Chatbot · Space Exploration & Astronomy

---

## 🗂️ Project Structure

```
chatbot/
├── backend/
│   ├── data/
│   │   └── space_knowledge.txt     # Knowledge base document
│   ├── chroma_db/                  # Auto-generated vector store
│   ├── main.py                     # Flask server (entry point)
│   ├── rag.py                      # RAG pipeline (embed + retrieve)
│   ├── nova.py                     # NOVA agent logic + 3 scenarios
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    └── index.html                  # Full chatbot UI (HTML/CSS/JS)
```

---

## ⚙️ Setup & Run

### 1. Clone / navigate to the project
```bash
cd chatbot/backend
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key
```bash
cp .env.example .env
# Edit .env and add your key:
# OPENAI_API_KEY=sk-...
```

### 5. Run the server
```bash
python main.py
```

### 6. Open in browser
```
http://localhost:5000
```

---

## 🤖 How the Agentic RAG Works

When a user sends a message, NOVA runs a 3-step pipeline:

```
User Input
    │
    ▼
┌─────────────────────────────────┐
│  STEP 1: Topic Classification   │  ← GPT-4o-mini decides domain
│  "Is this about space/astro?"   │
└────────────┬────────────────────┘
             │
      ┌──────┴──────┐
      │             │
   OUT OF          IN DOMAIN
   DOMAIN             │
      │               ▼
  Scenario 1   ┌─────────────────┐
  (introduce   │ STEP 2: RAG     │  ← ChromaDB similarity search
   NOVA)       │ Retrieve chunks │     (relevance score ≥ 0.3)
               └──────┬──────────┘
                      │
             ┌────────┴────────┐
             │                 │
          FOUND            NOT FOUND
             │                 │
             ▼                 ▼
        Scenario 2        Scenario 3
        (answer from      ("I don't know")
         context)
```

### Scenario Labels (shown in UI badge):
| Badge | Meaning |
|-------|---------|
| 🟡 OUT OF DOMAIN | Topic not related to space/astronomy |
| 🟢 RAG HIT | Found in knowledge base, answered |
| 🔴 RAG MISS | Space topic but not in our data |

---

## 🛠️ Customizing

- **Change the topic**: Replace `space_knowledge.txt` in `data/` with your own document, update the agent identity in `nova.py`
- **Tune retrieval**: Adjust `k` and the score threshold `0.3` in `rag.py`
- **Rebuild vector store**: Delete the `chroma_db/` folder and restart the server

---

## 📦 Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | HTML5 + CSS3 + Vanilla JS |
| Backend | Flask (Python) |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | text-embedding-3-small |
| Vector Store | ChromaDB (local) |
| RAG Framework | LangChain |
