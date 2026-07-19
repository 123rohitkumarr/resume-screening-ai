# AI Resume Screening & Candidate Feedback System

This is an end-to-end system I built to make resume screening smarter and less biased. Instead of just keyword-matching a resume against a job description, it actually looks at semantic meaning, skill overlap, experience, and education — and then explains *why* it gave the score it did, so recruiters aren't just trusting a black box.

It combines a few different pieces of ML/NLP tech: a trained XGBoost classifier for predicting job roles, sentence embeddings + FAISS for semantic search, fuzzy skill matching, and an LLM-powered chatbot that can talk through a candidate's resume and suggest how they could close skill gaps.

## What it does

- **Parses resumes** from PDF, DOCX, or plain text — no manual reformatting needed.
- **Scores candidates on four dimensions**: semantic fit, skill overlap, experience match, and education level — not just a single opaque number.
- **Predicts the most likely job role** for a resume using a trained XGBoost model.
- **Finds semantically similar resumes/roles fast** using Sentence-Transformer embeddings and a FAISS vector index.
- **Explains its scoring** — every score comes with a breakdown of what contributed to it, so it's not a black box.
- **Runs a RAG-based chatbot** that references the actual resume to give career coaching advice.
- **Auto-generates feedback** for candidates (why they were or weren't a good fit, plus resources to close gaps).
- **Works with multiple LLM providers** — Gemini, OpenAI, or a local Ollama instance, your choice.
- **Has a live dashboard** built in Streamlit for recruiters to explore results interactively.

## How it's put together

The project is split into two independent services — a FastAPI backend that does all the heavy lifting, and a Streamlit frontend that recruiters actually interact with. They talk to each other over a REST API, which means you could swap out the frontend entirely (say, for a React app) without touching the backend.

```
Streamlit Frontend  <---- REST API ---->  FastAPI Backend
  (what recruiters                        (all the ML/NLP,
   actually use)                           routing, and DB logic)
                                                  |
                        -------------------------------------------------
                        |              |                |               |
                   XGBoost         Embeddings       RAG Pipeline      SQLite
                  (role predictor)  (MiniLM-L6-v2)   (LLM chatbot)   (history/
                                         |                            feedback)
                                    FAISS Index
                                  (semantic search)
```

## Tech stack

| Layer | What's used |
|---|---|
| Backend | FastAPI, Python 3.11, Pydantic |
| Frontend | Streamlit, Pandas, Plotly |
| Machine learning | XGBoost, scikit-learn, RapidFuzz |
| NLP / embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Generative AI | LangChain, Gemini, OpenAI, Ollama |
| Vector search | FAISS |
| Database | SQLite via SQLAlchemy |

## Project layout

```
resume-screening-ai/
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── api/                 # route handlers
│   ├── core/                # config & security
│   ├── models/               # DB models
│   └── services/             # business logic + ML inference
├── frontend/
│   └── app.py                # the Streamlit app
├── ml/
│   ├── train_classifier.py   # trains the XGBoost model
│   └── build_embeddings.py   # builds the FAISS index
├── data/
│   ├── seed_data.py           # generates mock candidates
│   └── sample_resumes/
├── tests/
├── .env.example
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## Getting it running locally

**1. Clone and install dependencies**
```bash
git clone https://github.com/your-org/resume-screening-ai.git
cd resume-screening-ai
pip install -r requirements.txt
```

**2. Set up your environment variables**
```bash
cp .env.example .env
```
Then fill in whichever LLM provider's API key you plan to use.

**3. Generate some seed data to test with**
```bash
python data/seed_data.py
```

**4. Train the role classifier**
```bash
python ml/train_classifier.py
```

**5. Build the vector index**
```bash
python ml/build_embeddings.py
```

**6. Run both services** (you'll need two terminals since they're separate processes)
```bash
# terminal 1 — backend
uvicorn app.main:app --reload --port 8000

# terminal 2 — frontend
streamlit run frontend/app.py
```

The API lands on `http://localhost:8000`, the UI on `http://localhost:8501`.

**Running the tests**
```bash
python -m pytest tests/ -v
```

**Prefer Docker?**
```bash
docker-compose up --build
```
Same ports as above, just containerized.

## API endpoints

| Endpoint | What it does |
|---|---|
| `GET /health` | Health check — DB connection, model loading status |
| `POST /api/screening/` | Screens a candidate from raw JSON text |
| `POST /api/screening/file` | Uploads and parses a resume file, then screens it |
| `POST /api/feedback/` | Generates candidate feedback + learning resources via LLM |
| `POST /api/chat/` | Talks to the RAG career-coaching chatbot |
| `GET /api/chat/history/{session_id}` | Fetches chat history for a session |
| `DELETE /api/chat/history/{session_id}` | Clears chat history for a session |

## How the scoring actually works

The compatibility score isn't a black box — it's a weighted combination of four things:

1. **Semantic similarity (40%)** — the resume and job description both get embedded into vectors via Sentence-Transformers, and cosine similarity measures how contextually close they are.
2. **Skill match (30%)** — fuzzy matching (via RapidFuzz) compares extracted skills against required ones, so "React.js" and "React" aren't treated as a mismatch.
3. **Experience fit (15%)** — a Gaussian penalty curve rewards a close match to the required years of experience, while penalizing being significantly under- or over-qualified.
4. **Education (15%)** — an ordinal comparison checks whether the candidate meets, exceeds, or falls short of the required education level.

## Configuration

Everything's controlled through `.env`:

- `LLM_PROVIDER` — `openai`, `gemini`, or `ollama`
- `OPENAI_API_KEY` — needed if you're using OpenAI models
- `GEMINI_API_KEY` — needed if you're using Gemini
- `OLLAMA_BASE_URL` — for local Ollama (defaults to `http://localhost:11434`)
- `DB_PATH` — SQLite database path (defaults to `./data/screening.db`)
- `FAISS_INDEX_PATH` — where the vector index gets stored

## License

MIT — see `LICENSE` for the full text.
