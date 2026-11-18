@echo off
setlocal
title Stop Resume Screening AI

echo ===================================================
echo     Stopping Resume Screening AI Services...
echo ===================================================

echo [1/2] Terminating processes on port 8000 (FastAPI)...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8000" ^| findstr "LISTENING"') do (
    echo Stopping PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

echo [2/2] Terminating processes on port 8501 (Streamlit)...
for /f "tokens=5" %%a in ('netstat -aon 2^>nul ^| findstr ":8501" ^| findstr "LISTENING"') do (
    echo Stopping PID %%a...
    taskkill /F /PID %%a >nul 2>&1
)

echo ===================================================
echo     All services stopped successfully.
echo ===================================================
ping -n 3 127.0.0.1 >nul
exit /b 0
