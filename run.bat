@echo off
title Agentic AI - Weather Agent
echo.
echo  ⚡  Agentic AI - Smart Weather Agent
echo  ════════════════════════════════════
echo.
echo  Starting server...
echo  Open http://localhost:8000 in your browser
echo.
echo  Press Ctrl+C to stop the server.
echo.

cd /d "%~dp0"
python main.py

pause
