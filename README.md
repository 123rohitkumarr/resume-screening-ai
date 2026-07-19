<h1 align="center">🤖 AI Resume Screening & Candidate Feedback System</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/FastAPI-0.100+-green.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-1.25+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</p>

## 🌟 Overview
A production-ready, end-to-end AI system designed to streamline and revolutionize the recruitment process. Utilizing modern Machine Learning, Deep Learning, NLP, RAG (Retrieval-Augmented Generation), and Explainable AI, this application provides unbiased, multi-dimensional resume analysis, job-role classification, and personalized candidate feedback.

## ✨ Key Features
- **Resume Parsing:** Robust text extraction from multiple formats including PDF, DOCX, and TXT files.
- **Multi-Dimensional Compatibility Scoring:** Evaluates candidate fit across semantic context, explicit skills, experience duration, and educational background.
- **XGBoost Job-Role Classification:** Accurately categorizes resumes into appropriate job families or roles based on learned historical patterns.
- **Advanced Vector Search:** Leverages Sentence-Transformer embeddings (`all-MiniLM-L6-v2`) and FAISS for lightning-fast semantic resume matching.
- **Explainable AI (XAI):** Provides transparent score attribution so recruiters know exactly *why* a candidate received a specific score.
- **RAG-Powered Chatbot:** Interactive career coaching assistant that references the candidate's actual resume to offer personalized advice.
- **AI Feedback Generator:** Automatically crafts constructive rejection/acceptance feedback, along with targeted learning resources for skill gaps.
- **Multi-LLM Support:** Seamlessly swap between Gemini, Ollama (local), and OpenAI models for generative tasks.
- **Real-Time Analytics Dashboard:** Streamlit-powered interface for interactive exploration, filtering, and reporting.

## 🏗️ Architecture
The system employs a decoupled microservices-inspired architecture, separating the heavy ML processing and APIs from the user interface.

```text
+---------------------+         REST API          +-----------------------+
|  Streamlit Frontend | <=======================> |    FastAPI Backend    |
|  (Recruiter/Admin)  |                           |  (Core Logic & APIs)  |
+---------------------+                           +-----------+-----------+
                                                              |
          +-------------------------+-------------------------+-------------------------+
          |                         |                         |                         |
          v                         v                         v                         v
+-------------------+     +-------------------+     +-------------------+     +-------------------+
| XGBoost Classifier|     |  Embeddings Model |     |     RAG Pipeline  |     | SQLite Database   |
| (Job Prediction)  |     |(all-MiniLM-L6-v2) |     |  (LLM / Chatbot)  |     | (Feedback/History)|
+-------------------+     +---------+---------+     +-------------------+     +-------------------+
                                    |
                                    v
                          +-------------------+
                          | FAISS Vector DB   |
                          | (Semantic Search) |
                          +-------------------+
```

## 🛠️ Tech Stack

| Category | Technology Stack |
| :--- | :--- |
| **Backend Framework** | FastAPI, Python 3.11, Pydantic |
| **Frontend UI** | Streamlit, Pandas, Plotly |
| **Machine Learning** | XGBoost, scikit-learn, RapidFuzz |
| **NLP & Embeddings** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Generative AI & LLMs**| LangChain, Google Gemini, OpenAI, Ollama |
| **Vector Database** | FAISS (Facebook AI Similarity Search) |
| **Relational Database** | SQLite (with SQLAlchemy ORM) |

## 📁 Project Structure
```text
resume-screening-ai/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── api/                 # API routers and endpoints
│   ├── core/                # Configuration and security
│   ├── models/              # Database models
│   └── services/            # Business logic and ML inference
├── frontend/
│   └── app.py               # Streamlit application
├── ml/
│   ├── train_classifier.py  # Script to train the XGBoost model
│   └── build_embeddings.py  # Script to generate FAISS indices
├── data/
│   ├── seed_data.py         # Mock data generator
│   └── sample_resumes/      # Test documents
├── tests/                   # Pytest test suite
├── .env.example             # Template for environment variables
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Docker orchestration
└── README.md                # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- **Python:** 3.11 or higher
- **pip:** Up to date
- **Docker:** (Optional, for containerized deployment)

### Installation
```bash
git clone https://github.com/your-org/resume-screening-ai.git
cd resume-screening-ai
pip install -r requirements.txt
```

### Setup
1. **Configure Environment Variables:**
   Copy the example config and add your API keys.
   ```bash
   cp .env.example .env
   ```
2. **Generate Seed Data:**
   Create synthetic candidate data and resumes for testing.
   ```bash
   python data/seed_data.py
   ```
3. **Train the Job-Role Classifier:**
   ```bash
   python ml/train_classifier.py
   ```
4. **Build the Vector Store:**
   Generate embeddings and the FAISS index.
   ```bash
   python ml/build_embeddings.py
   ```

### Running the Application
To run the system locally without Docker, you will need two terminal instances:

**1. Start the Backend (API):**
```bash
uvicorn app.main:app --reload --port 8000
```

**2. Start the Frontend (UI):**
```bash
streamlit run frontend/app.py
```

**3. Run Tests:**
```bash
python -m pytest tests/ -v
```

### 🐳 Docker Deployment
You can spin up the entire application stack using Docker Compose:
```bash
docker-compose up --build
```
*The API will be available at `http://localhost:8000` and the UI at `http://localhost:8501`.*

## 🔌 API Endpoints

The backend provides a robust REST API for seamless integration with other internal tools.

- `GET /health` : Checks system health, DB connection, and model loading status.
- `POST /api/screening/` : Screens a candidate using a JSON payload containing raw text and extracted features.
- `POST /api/screening/file` : Uploads a document (PDF/DOCX/TXT), parses it, and returns a detailed screening analysis.
- `POST /api/feedback/` : Generates detailed, constructive candidate feedback and learning resources via LLM.
- `POST /api/chat/` : Interacts with the RAG-powered career coaching chatbot.
- `GET /api/chat/history/{session_id}` : Retrieves chat history for a specific career coaching session.
- `DELETE /api/chat/history/{session_id}` : Clears the chat history for a given session.

## 🧮 Scoring Algorithm

Our compatibility score is deterministic, explainable, and multi-dimensional. It weights four primary criteria:

1. **Semantic Similarity (40%):** Uses Sentence-Transformers to convert both the resume and the job description into dense vector embeddings. Calculates the cosine similarity to assess overall contextual fit.
2. **Skill Match (30%):** Utilizes `rapidfuzz` for fuzzy string matching between extracted candidate skills and required job skills, handling variations in terminology (e.g., "React.js" vs "React").
3. **Experience Fit (15%):** Applies a Gaussian penalty model. Exact matches or slight over-qualification receive top marks, while significant under-qualification or severe over-qualification are penalized smoothly.
4. **Education (15%):** Ordinal comparison logic. Checks if the candidate meets, exceeds, or falls short of the baseline educational requirement (e.g., Bachelor's vs Master's).

## ⚙️ Configuration

The `.env` file controls system behavior and model integrations. Key variables include:

- `LLM_PROVIDER`: Choose between `openai`, `gemini`, or `ollama`.
- `OPENAI_API_KEY`: Required if using OpenAI models (e.g., GPT-4o).
- `GEMINI_API_KEY`: Required if using Google's Gemini models.
- `OLLAMA_BASE_URL`: The URL for local Ollama instances (default: `http://localhost:11434`).
- `DB_PATH`: Path to the SQLite database (default: `./data/screening.db`).
- `FAISS_INDEX_PATH`: Path to the compiled FAISS vector store.

## 📄 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.
