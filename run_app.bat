@echo off
cd /d "%~dp0"
title CT Head AI -- Classical Radiomics Triage Dashboard
echo ==============================================================================
echo   CT Head AI -- Launching Interactive Streamlit Clinical Dashboard
echo   Delhi Technological University (DTU) - Classical ML Capstone
echo ==============================================================================
echo Starting local web server on http://localhost:8501 ...
echo.
"C:\Users\HARSH\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run app.py --server.port=8501 --server.headless=false
pause
