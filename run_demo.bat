@echo off
cd /d "%~dp0"
echo ==============================================================================
echo   CT Head AI -- Launching Clinical Triage Demo
echo ==============================================================================
"C:\Users\HARSH\AppData\Local\Programs\Python\Python311\python.exe" demo.py --case data/sample_cases/case_hemorrhage.npy
pause
