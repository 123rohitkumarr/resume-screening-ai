import os
from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.config import settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Verify the presence of a valid API key if configured.
    If no API_KEY is set in settings or environment, development/demo mode is active and open.
    In production with API_KEY set, requests without valid X-API-Key are rejected with 401.
    """
    configured_key = getattr(settings, "api_key", None) or os.getenv("API_KEY", "")
    if configured_key and api_key != configured_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key (use X-API-Key header)",
        )
    return api_key
