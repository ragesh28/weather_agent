@echo off
title SkyStream AI - Weather Agent
echo.
echo  ⚡  SkyStream AI - Smart Weather Agent
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
