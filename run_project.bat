@echo off
setlocal
title Resume Screening AI - Launcher

echo ================================================================
echo           AI Resume Screening and Candidate System
echo ================================================================
echo.

:: 1. Navigate to script directory
cd /d "%~dp0"
echo [INFO] Working Directory: %CD%

:: 2. Check Python installation
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in your PATH.
    echo Please install Python 3.10+ and ensure it is added to your PATH.
    pause
    exit /b 1
)

:: 3. Detect or activate Virtual Environment
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment in .venv
    call .venv\Scripts\activate.bat
    goto :venv_ready
)

if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment in venv
    call venv\Scripts\activate.bat
    goto :venv_ready
)

echo [INFO] Using installed Python environment.

:venv_ready

:: 4. Check Environment Configuration file
if not exist ".env" (
    echo [INFO] Copying .env from .env.example
    copy ".env.example" ".env" >nul
)

:: 5. Verify core packages
echo [INFO] Verifying core Python packages...
python -c "import fastapi, uvicorn, streamlit, xgboost, sentence_transformers, faiss, rapidfuzz" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [INFO] Installing required dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Package installation failed. Please check your internet connection.
        pause
        exit /b 1
    )
) else (
    echo [INFO] All core dependencies verified.
)

:: 6. Verify and generate training data and models if missing
if not exist "data\models\classifier.joblib" (
    echo [INFO] Classifier model not found. Running training pipeline...
    if not exist "data\raw\resume_dataset.json" (
        echo [INFO] Generating seed dataset...
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
echo [INFO] Checking for existing instances on ports 8000 and 8501...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo [INFO] Clearing port 8000 - PID %%a
    taskkill /F /PID %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo [INFO] Clearing port 8501 - PID %%a
    taskkill /F /PID %%a >nul 2>&1
)

:: 8. Create logs directory
if not exist "logs" mkdir logs

:: 9. Start FastAPI Backend in a separate window
echo [INFO] Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "FastAPI Backend" cmd /k "cd /d "%~dp0" && if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"

:: 10. Wait briefly for backend startup
ping -n 4 127.0.0.1 >nul

:: 11. Start Streamlit Frontend in a separate window
echo [INFO] Starting Streamlit Frontend on http://localhost:8501 ...
start "Streamlit Frontend" cmd /k "cd /d "%~dp0" && if exist .venv\Scripts\activate.bat (call .venv\Scripts\activate.bat) && python -m streamlit run frontend\app.py --server.port 8501"

:: 12. Display summary
echo.
echo ================================================================
echo           Services Started Successfully!
echo ================================================================
echo.
echo   * Streamlit Frontend:   http://localhost:8501
echo   * FastAPI Backend:      http://127.0.0.1:8000
echo   * Swagger API Docs:     http://127.0.0.1:8000/docs
echo   * Health Endpoint:      http://127.0.0.1:8000/health
echo.
echo   To stop all running services, run: stop_project.bat
echo ================================================================
echo.

:: Open default browser
ping -n 3 127.0.0.1 >nul
start http://localhost:8501

echo Press any key to exit this launcher window. Services will stay running.
pause >nul
exit /b 0
