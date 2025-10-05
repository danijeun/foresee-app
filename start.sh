#!/bin/bash

echo "========================================"
echo "Starting Foresee App"
echo "========================================"
echo ""

# Activate virtual environment
echo "[1/2] Activating Python virtual environment..."
source myenv/bin/activate

# Start backend in background
echo "[2/2] Starting backend server..."
cd backend
python app.py &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start frontend
echo "Starting frontend..."
cd frontend
npm run dev

# Cleanup: kill backend when frontend stops
kill $BACKEND_PID 2>/dev/null
