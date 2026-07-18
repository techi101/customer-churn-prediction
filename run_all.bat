@echo off
set PYTHONIOENCODING=utf-8
echo.
echo ============================================================
echo   Customer Churn Prediction -- Full Pipeline Runner
echo ============================================================
echo.

echo [1/3] Generating synthetic dataset...
py -3 data_generator.py
if %errorlevel% neq 0 ( echo ERROR in data_generator.py & pause & exit /b )

echo.
echo [2/3] Training models and generating reports...
py -3 churn_model.py
if %errorlevel% neq 0 ( echo ERROR in churn_model.py & pause & exit /b )

echo.
echo [3/3] Building executive narrative report...
py -3 report_generator.py
if %errorlevel% neq 0 ( echo ERROR in report_generator.py & pause & exit /b )

echo.
echo ============================================================
echo   ALL DONE! Launching Dashboard...
echo   Open your browser at: http://localhost:8501
echo ============================================================
echo.
py -3 -m streamlit run dashboard.py
