import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.database import init_db, SessionLocal
from app.models.schemas import HealthCheck
from app.routers import screening, feedback, chatbot, settings_router
from app.utils.auth import verify_api_key
from app.utils.helpers import setup_logger, ensure_dir
from app.services.classifier import classifier
from app.services.embeddings import embedding_engine

log = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan event handler for startup and shutdown initialization."""
    log.info("Starting up API and checking resources...")

    # Ensure required data and model directories exist
    ensure_dir(settings.data_dir)
    ensure_dir(settings.raw_data_dir)
    ensure_dir(settings.models_dir)
    ensure_dir(settings.processed_dir)

    # Initialize SQL database
    try:
        init_db()
        log.info("Database initialized successfully.")
    except Exception as e:
        log.error(f"Database initialization error: {e}")

    # Load ML classifier
    classifier_loaded = classifier._load_models()
    if classifier_loaded:
        log.info("Job classifier loaded successfully.")
    else:
        log.warning("Job classifier could not be loaded. Please train the model using ml/train_classifier.py.")

    # Load FAISS index
    try:
        embedding_engine.load_index()
        log.info("FAISS vector store loaded successfully.")
    except Exception as e:
        log.warning(f"Could not load FAISS vector store index: {str(e)}")

    yield

    log.info("Shutting down API...")


# Initialize FastAPI App with Lifespan
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Production-ready, end-to-end AI resume screening API.",
    debug=settings.debug,
    lifespan=lifespan,
)

# CORS configuration
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
if not origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(screening.router, dependencies=[Depends(verify_api_key)])
app.include_router(feedback.router, dependencies=[Depends(verify_api_key)])
app.include_router(chatbot.router, dependencies=[Depends(verify_api_key)])
app.include_router(settings_router.router, dependencies=[Depends(verify_api_key)])


@app.get("/health", response_model=HealthCheck)
def health_check():
    """Check application status, database connectivity, and ML model availability."""
    if classifier.model is None:
        classifier._load_models()
    if embedding_engine.index is None:
        try:
            embedding_engine.load_index()
        except Exception:
            pass

    db_ok = False
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        db_ok = True
    except Exception:
        db_ok = False

    models_loaded = {
        "classifier": classifier.model is not None,
        "tfidf": classifier.tfidf is not None,
        "faiss_index": embedding_engine.index is not None,
        "database": db_ok,
    }
    return HealthCheck(
        status="healthy",
        version=settings.version,
        models_loaded=models_loaded,
    )
