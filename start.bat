@echo off
echo ========================================
echo Starting Foresee App
echo ========================================
echo.

REM Activate virtual environment
echo [1/2] Activating Python virtual environment...
call myenv\Scripts\activate.bat

REM Start backend in background
echo [2/2] Starting backend server...
start "Foresee Backend" cmd /k "cd backend && python app.py"

REM Wait a bit for backend to start
timeout /t 3 /nobreak > nul

REM Start frontend
echo Starting frontend...
cd frontend
npm run dev

pause
