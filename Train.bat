@echo off
REM === Navigate to the folder of this script ===
cd /d "%~dp0"

REM pip install flask-cors

REM === (Optional) activate virtual environment ===
REM call venv\Scripts\activate

REM === Run the Python web server ===
python TrainModel.py

REM === Keep window open so you can see logs ===
pause