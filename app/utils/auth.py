import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from dotenv import load_dotenv

# Optional local load
load_dotenv()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Simple API Key authentication. Can be set in environment variables.
# If no key is set in .env, API is open by default for easier development/demo.
API_KEY = os.getenv("API_KEY", "")

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify the presence of a valid API key if configured.
    """
    if API_KEY and api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key
