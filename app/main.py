import os
from fastapi import FastAPI, Depends, Security
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models.database import init_db
from app.models.schemas import HealthCheck
from app.routers import screening, feedback, chatbot
from app.utils.auth import verify_api_key
from app.utils.helpers import setup_logger, ensure_dir
from app.services.classifier import classifier
from app.services.embeddings import embedding_engine

log = setup_logger(__name__)

# Initialize FastAPI App
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Production-ready, end-to-end AI resume screening API.",
    debug=settings.debug
)

# CORS configuration
origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global API Key Security Dependency (can be activated selectively)
# For the demo, we mount endpoints and can secure them using verify_api_key if desired.
# For general routing:
app.include_router(screening.router, dependencies=[Depends(verify_api_key)])
app.include_router(feedback.router, dependencies=[Depends(verify_api_key)])
app.include_router(chatbot.router, dependencies=[Depends(verify_api_key)])

@app.on_event("startup")
def on_startup():
    log.info("Starting up API and checking resources...")
    # Make directories
    ensure_dir(settings.data_dir)
    ensure_dir(settings.raw_data_dir)
    ensure_dir(settings.models_dir)
    ensure_dir(settings.processed_dir)

    # Initialize SQL database
    init_db()
    log.info("Database initialized.")

    # Try loading ML classifier
    classifier_loaded = classifier._load_models()
    if classifier_loaded:
        log.info("Job classifier loaded successfully.")
    else:
        log.warning("Job classifier could not be loaded. Please train the model.")

    # Try loading FAISS index
    try:
        embedding_engine.load_index()
        log.info("FAISS vector store loaded successfully.")
    except Exception as e:
        log.warning(f"Could not load FAISS vector store index: {str(e)}")

@app.get("/health", response_model=HealthCheck)
def health_check():
    """
    Check application status and model availability.
    """
    models_loaded = {
        "classifier": classifier.model is not None,
        "tfidf": classifier.tfidf is not None,
        "faiss_index": embedding_engine.index is not None,
    }
    return HealthCheck(
        status="healthy",
        version=settings.version,
        models_loaded=models_loaded
    )
