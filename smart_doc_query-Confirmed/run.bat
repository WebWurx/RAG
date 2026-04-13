@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Starting Smart Doc Query...
echo Open your browser and go to: http://127.0.0.1:5000
echo.
python app.py
pause
