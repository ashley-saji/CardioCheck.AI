@echo off
echo Starting Heart Disease Prediction Web App...
echo.

cd /d "C:\Users\ashle\OneDrive\Desktop\web app final"

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting Streamlit server...
echo.
echo The app will open in your default browser.
echo Press Ctrl+C to stop the server.
echo.

streamlit run streamlit_app.py

pause