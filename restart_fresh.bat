@echo off
echo ========================================
echo TFET Optimization - Fresh Start
echo ========================================
echo.

REM Kill any existing Python processes
taskkill /F /IM python.exe 2>nul

REM Wait a moment
timeout /t 2 /nobreak >nul

echo Starting Flask application with NEW graph settings...
echo.
echo IMPORTANT: After server starts:
echo 1. Open browser to http://localhost:5000
echo 2. Press Ctrl+Shift+Delete to clear cache
echo 3. Or press Ctrl+F5 for hard refresh
echo 4. Run a NEW optimization to generate new graphs
echo.
echo ========================================
echo.

cd tfet_optimization_agent\web_interface
python app.py

pause
