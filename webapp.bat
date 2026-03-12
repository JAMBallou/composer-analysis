@echo off
REM run_webapp.bat
REM Quick launcher for the Composer Classification Web App

echo ============================================================
echo Composer Classification Web App
echo ============================================================
echo.

REM Check if virtual environment exists
if exist .venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo Warning: Virtual environment not found at .venv
    echo Using system Python...
)

echo.
echo Starting Flask server...
echo.
echo Open your browser to: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python web\app.py

pause
