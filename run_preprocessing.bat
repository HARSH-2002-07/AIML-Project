@echo off
echo ==============================================================================
echo   CT Head AI -- Launching Preprocessing Visualizer
echo ==============================================================================
"C:\Users\HARSH\AppData\Local\Programs\Python\Python311\python.exe" visualize_preprocessing.py --case data/sample_cases/case_hemorrhage.npy
echo.
echo Visual diagnostic figure saved to reports/preprocessing_pipeline_visualization.png
pause
