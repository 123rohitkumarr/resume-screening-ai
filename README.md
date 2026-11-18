# AI Resume Screening & Candidate Feedback System

An end-to-end AI-powered resume screening, scoring, explainability, and career assistance platform. Instead of simple keyword matching, this system combines deep semantic similarity, fuzzy skill ontology mapping, experience gap modeling, and ordinal education analysis to calculate unbiased, explainable compatibility scores.

It features a trained **XGBoost** role classifier, **Sentence-Transformers (`all-MiniLM-L6-v2`)** embeddings with **FAISS** vector indexing, and a **Retrieval-Augmented Generation (RAG)** career coaching assistant.

---

## What It Does

- **Multi-Format Resume Parsing**: Supports PDF (via PyMuPDF), DOCX (paragraphs + tables), and plain text with automatic file size, corruption, and empty document validation.
- **Explainable 4-Dimension Candidate Scoring**:
  - **Semantic Similarity (40%)**: Cosine distance using dense sentence embeddings.
  - **Skill Matching (30%)**: Fuzzy skill matching with preferred bonus weighting.
  - **Experience Matching (15%)**: Gaussian curve rewarding exact experience fits.
  - **Education Matching (15%)**: Ordinal comparison without qualification inflation.
- **XGBoost Job Role Classifier**: Predicts target professional roles from combined skill TF-IDF features, one-hot encoded education, and scaled years of experience.
- **FAISS Semantic Search**: Retrieves relevant profile vectors and candidate benchmarks with cross-candidate privacy protection.
- **Explainable AI (XAI)**: SHAP-style breakdown explaining positive, neutral, and negative factors contributing to candidate evaluation.
- **Contextual RAG Chatbot**: Grounded conversational career advisor that generates tailored interview questions and preparation roadmaps without hallucinating private qualifications.
- **Dynamic Configuration**: Full runtime adjustment of scoring weights (must sum to 100%) and LLM settings synchronized between frontend and backend.

---

## Architecture

The system is organized into two independent, decoupled services connected via REST API:

```
Streamlit Web UI (Port 8501)  <==== REST API ====>  FastAPI Backend Engine (Port 8000)
    ├── Dashboard (Upload & Match)                           ├── /api/screening/
    ├── Chatbot (RAG Assistant)                              ├── /api/chat/
    ├── Analytics (Distributions)                            ├── /api/feedback/
    └── Settings (Runtime Weights)                           ├── /api/settings/
                                                             └── /health
                                                                    │
                     ┌──────────────────────────────────────────────┴───────────────────────────┐
                     ▼                                              ▼                           ▼
          Machine Learning (XGBoost)                   Embeddings & Vector Index          SQLite Database
     (TF-IDF + OHE Edu + Scaled Exp -> Role)             (all-MiniLM-L6-v2 + FAISS)    (Resumes, JDs, Screenings)
```

---

## Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | FastAPI, Python 3.10+, Pydantic v2, Pydantic-Settings |
| **Frontend** | Streamlit, Plotly, Pandas |
| **Machine Learning** | XGBoost, scikit-learn, RapidFuzz |
| **Deep Learning & NLP** | Sentence-Transformers (`all-MiniLM-L6-v2`), NLTK |
| **Vector Search** | FAISS CPU (`IndexFlatIP`) |
| **Document Parsing** | PyMuPDF (`fitz`), python-docx |
| **Database** | SQLite via SQLAlchemy 2.0 |
| **Generative AI** | Google Gemini (`gemini-2.0-flash`), OpenAI, local Ollama |

---

## Project Structure

```
resume-screening-ai/
├── app/
│   ├── config.py             # Central Pydantic settings & weight validation
│   ├── main.py               # FastAPI application entry point with lifespan
│   ├── models/
│   │   ├── database.py       # SQLAlchemy ORM models (resumes, screenings, chats)
│   │   └── schemas.py        # Pydantic request & response schemas
│   ├── routers/
│   │   ├── chatbot.py        # RAG chat & message history routes
│   │   ├── feedback.py       # Candidate feedback generation routes
│   │   ├── screening.py      # Resume text & file screening routes
│   │   └── settings_router.py# Runtime weights & LLM settings synchronization
│   ├── services/
│   │   ├── classifier.py     # XGBoost inference (505-dim feature transform)
│   │   ├── embeddings.py     # SentenceTransformers & FAISS index search
│   │   ├── explainer.py      # XAI attribution & improvement suggestions
│   │   ├── feedback_generator.py # Skill gap & learning resource recommendations
│   │   ├── matcher.py        # 4-dimension composite scoring engine
│   │   ├── parser.py         # PDF, DOCX (with tables), and TXT parser
│   │   ├── preprocessor.py   # Name, email, phone, experience & edu extraction
│   │   ├── rag_engine.py     # Contextual RAG with privacy safeguards
│   │   └── skill_extractor.py# Regex boundary matching & skill ontology
│   └── utils/
│       ├── auth.py           # Optional API key verification (X-API-Key)
│       └── helpers.py        # Logging, text cleaning & clamping utilities
├── data/
│   ├── seed_data.py          # Synthetic candidate dataset generator
│   ├── models/               # Serialized models (classifier, tfidf, faiss)
│   └── raw/                  # Raw dataset files
├── frontend/
│   ├── app.py                # Streamlit landing page & system health status
│   ├── utils.py              # Shared API client & session-state management
│   ├── components/           # UI components (gauges, radar, XAI cards)
│   ├── pages/                # Multi-page views (Dashboard, Chatbot, Analytics, Settings)
│   └── styles/theme.py       # Custom modern dark theme styling
├── ml/
│   ├── build_embeddings.py   # Builds FAISS vector store from dataset
│   ├── evaluate.py           # Comprehensive cross-validation & evaluation
│   └── train_classifier.py   # Trains XGBoost role classifier
├── tests/
│   ├── test_api.py           # API endpoint tests
│   ├── test_end_to_end.py    # Complete integration & file upload tests
│   ├── test_matcher.py       # Scoring dimension unit tests
│   ├── test_preprocessor.py  # Extraction unit tests
│   └── test_skill_extractor.py# Skill ontology unit tests
├── .env.example              # Environment variables template
├── requirements.txt          # Python package dependencies
├── run_project.bat           # 1-Click Windows automatic startup script
├── stop_project.bat          # 1-Click Windows automatic shutdown script
└── README.md
```

---

## Localhost Quickstart (Windows)

### Option 1: 1-Click Startup (Recommended)
Simply double-click or run:
```cmd
run_project.bat
```
This script automatically:
1. Detects and activates your Python virtual environment (`.venv`).
2. Checks and copies `.env` if missing.
3. Verifies required dependencies.
4. Generates data and trains the XGBoost model if model files are missing.
5. Clears any stale processes on ports `8000` and `8501`.
6. Launches FastAPI on `http://127.0.0.1:8000` and Streamlit on `http://localhost:8501`.
7. Opens `http://localhost:8501` in your default browser.

To stop the application at any time, run:
```cmd
stop_project.bat
```

---

### Option 2: Manual Setup

1. **Create and activate a virtual environment**:
```cmd
python -m venv .venv
call .venv\Scripts\activate.bat
```

2. **Install dependencies**:
```cmd
python -m pip install -r requirements.txt
```

3. **Configure environment variables**:
```cmd
copy .env.example .env
```
*(Optional: add your `GEMINI_API_KEY` from Google AI Studio to enable live conversational AI. Otherwise, the system runs smoothly in grounded offline mode).*

4. **Train classifier and build vector index**:
```cmd
python data/seed_data.py
python ml/train_classifier.py
python ml/build_embeddings.py
```

5. **Run the services**:
- **Terminal 1 (Backend)**:
  ```cmd
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
  ```
- **Terminal 2 (Frontend)**:
  ```cmd
  python -m streamlit run frontend/app.py --server.port 8501
  ```

---

## Running the Automated Test Suite

Execute the complete 43-test suite covering file parsing, model inference, API routes, scoring, and XAI:
```cmd
python -m pytest tests/ -v
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Application status, database connection, and loaded models |
| `POST` | `/api/screening/` | Screen resume text against job requirements |
| `POST` | `/api/screening/file` | Upload and screen resume file (`.pdf`, `.docx`, `.txt`) |
| `POST` | `/api/feedback/` | Generate improvement feedback and course suggestions |
| `POST` | `/api/chat/` | Converse with RAG career coaching assistant |
| `GET` | `/api/chat/history/{session_id}` | Fetch chat history for session |
| `DELETE` | `/api/chat/history/{session_id}` | Clear chat history for session |
| `GET` | `/api/settings/` | Get active scoring weights and LLM provider |
| `PUT` | `/api/settings/` | Update runtime scoring weights (validated to sum to 1.0) |

Swagger UI documentation is accessible at **`http://127.0.0.1:8000/docs`**.

---

## Troubleshooting

- **Port Conflict (`WinError 10048`)**: Run `stop_project.bat` to terminate any zombie processes occupying ports 8000 or 8501.
- **NLTK Download Offline**: The preprocessor includes automatic fallbacks so missing NLTK internet access will not crash the app.
- **LLM Key Missing**: If no `GEMINI_API_KEY` is provided in `.env`, the RAG chatbot operates in grounded offline mode, answering questions directly from the candidate's screening results and skill gaps.

---

## License

MIT License.
