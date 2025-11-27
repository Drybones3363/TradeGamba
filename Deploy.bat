@echo off
REM === Navigate to the folder of this script ===
cd /d "%~dp0"

rmdir static
mkdir static
copy *.html static
copy *.js static

REM pip install flask-cors

REM === (Optional) activate virtual environment ===
REM call venv\Scripts\activate

REM === Run the Python web server ===
start http://localhost:5000
python AIWebServer.py

REM === Keep window open so you can see logs ===
pause
