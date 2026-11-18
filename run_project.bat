@echo off
setlocal enabledelayedexpansion
title Resume Screening AI - Startup

echo ================================================================
echo           AI Resume Screening & Candidate System
echo ================================================================
echo.

:: 1. Navigate to script directory
cd /d "%~dp0"
echo [INFO] Working Directory: %CD%

:: 2. Check Python installation
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not found in your PATH. Please install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

:: 3. Detect or create Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Found existing virtual environment (.venv). Activating...
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    echo [INFO] Found existing virtual environment (venv). Activating...
    call venv\Scripts\activate.bat
) else (
    echo [INFO] No virtual environment found. Creating .venv...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [WARNING] Failed to create virtual environment. Continuing with system Python...
    ) else (
        echo [INFO] Virtual environment created successfully.
        call .venv\Scripts\activate.bat
    )
)

:: 4. Check Environment Configuration file
if not exist ".env" (
    echo [INFO] .env file not found. Copying from .env.example...
    copy ".env.example" ".env" >nul
)

:: 5. Verify & Install dependencies
echo [INFO] Verifying core packages...
python -c "import fastapi, uvicorn, streamlit, xgboost, sentence_transformers, faiss, rapidfuzz" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing required dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Dependency installation encountered errors.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Core dependencies verified.
)

:: 6. Verify and generate training data and models if missing
if not exist "data\models\classifier.joblib" (
    echo [INFO] Classifier model not found. Running training pipeline...
    if not exist "data\raw\resume_dataset.json" (
        echo [INFO] Generating synthetic seed data...
        python data\seed_data.py
    )
    echo [INFO] Training XGBoost role classifier...
    python ml\train_classifier.py
)

if not exist "data\models\faiss_index" (
    echo [INFO] FAISS vector index not found. Building vector index...
    python ml\build_embeddings.py
)

:: 7. Free ports 8000 & 8501 if previously occupied
echo [INFO] Checking for stale service instances on ports 8000 and 8501...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [INFO] Stopping stale backend on port 8000 (PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo [INFO] Stopping stale frontend on port 8501 (PID %%a)...
    taskkill /F /PID %%a >nul 2>&1
)

:: 8. Create logs directory
if not exist "logs" mkdir logs

:: 9. Start FastAPI Backend in background
echo [INFO] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Resume Screening Backend (FastAPI)" cmd /k "title FastAPI Backend && cd /d "%~dp0" && if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) else if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

:: 10. Wait for backend to initialize
echo [INFO] Waiting for FastAPI to initialize...
timeout /t 4 /nobreak >nul

:: 11. Start Streamlit Frontend
echo [INFO] Starting Streamlit Frontend on http://localhost:8501 ...
start "Resume Screening Frontend (Streamlit)" cmd /k "title Streamlit Frontend && cd /d "%~dp0" && if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) else if exist venv\Scripts\activate.bat (call venv\Scripts\activate.bat) && python -m streamlit run frontend\app.py --server.port 8501 --server.headless false"

:: 12. Display summary and status
echo.
echo ================================================================
echo           Services Started Successfully!
echo ================================================================
echo.
echo   * Streamlit Frontend:   http://localhost:8501
echo   * FastAPI Backend:      http://127.0.0.1:8000
echo   * API Documentation:    http://127.0.0.1:8000/docs
echo   * Health Check:         http://127.0.0.1:8000/health
echo.
echo   To stop all services at any time, run: stop_project.bat
echo ================================================================
echo.

:: Open browser automatically
timeout /t 3 /nobreak >nul
start http://localhost:8501

echo Press any key to exit this launcher window (services will stay running).
pause >nul
exit /b 0
