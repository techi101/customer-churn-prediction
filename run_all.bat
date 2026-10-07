@echo off
REM WHAT THIS FILE IS: a Windows "one-click" runner. Double-click it (or type run_all.bat) in the project folder
REM and it runs the whole project in order: train the models, write the executive report, open the dashboard.
REM A .bat file is a list of Command Prompt commands run one after another. REM lines are comments (ignored).
REM Analogy: a recipe card that says "step 1 cook, step 2 plate, step 3 serve" so nobody has to remember the order.
REM Real example: step 1 prints lines like "LightGBM  Test AUC=0.8642" and writes reports/metrics.json,
REM step 2 writes reports/executive_report.md, step 3 opens the dashboard at http://localhost:8501.
REM Overall flow: churn_model.py, then report_generator.py, then streamlit run dashboard.py (stop early on any error).
REM
REM The first line, @echo off, stops Command Prompt from printing every command before running it,
REM so the window only shows the messages below. The @ hides that line itself too.
REM
REM Make Python print text as UTF-8. Without this, printing the rupee sign in report_generator.py can crash
REM on a Windows console (the README mentions the same fix).
set PYTHONIOENCODING=utf-8
REM "echo." prints an empty line. The other echo lines print a banner.
echo.
echo ============================================================
echo   Customer Churn Prediction -- Full Pipeline Runner
echo ============================================================
echo.

REM Step 1: train the 3 models and write models/, reports/ and data/scored_customers.csv.
REM "py -3" = the Windows Python launcher, asked for Python 3.
echo [1/2] Training models and generating reports...
py -3 churn_model.py
REM errorlevel = the exit code of the last program (0 = success). If it is not 0, print an error,
REM pause (wait for a key so the user can read the message), then exit /b (leave this script).
REM The whole if-block is on one line in brackets, so nothing may be inserted inside it.
if %errorlevel% neq 0 ( echo ERROR in churn_model.py & pause & exit /b )

REM Step 2: build the executive briefing from the files step 1 wrote (same error check after it).
echo.
echo [2/2] Building executive narrative report...
py -3 report_generator.py
if %errorlevel% neq 0 ( echo ERROR in report_generator.py & pause & exit /b )

REM Step 3: start the Streamlit dashboard. "python -m streamlit" runs Streamlit through this Python.
REM It keeps running (serving the web page) until you close the window or press Ctrl+C.
echo.
echo ============================================================
echo   ALL DONE! Launching Dashboard...
echo   Open your browser at: http://localhost:8501
echo ============================================================
echo.
py -3 -m streamlit run dashboard.py
