import sys
import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from app.main import app
except ImportError:
    # Dummy app in case main app is not implemented yet
    from fastapi import FastAPI
    app = FastAPI()

@pytest.fixture
def app_client():
    return TestClient(app)

@pytest.fixture
def mock_embedding_engine():
    # Mocking sentence_transformers to avoid downloading models during tests
    with patch('sentence_transformers.SentenceTransformer') as MockModel:
        instance = MockModel.return_value
        instance.encode.return_value = [[0.1] * 384] # Assuming 384-dimensional embeddings
        yield instance
