@echo off
echo ============================================
echo   ASL Platform Setup
echo ============================================
echo.

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo [2/3] Installing dependencies...
pip install opencv-python mediapipe tensorflow scikit-learn numpy flask gtts pygame pyttsx3 deep-translator protobuf pillow

echo [3/3] Done!
echo.
echo ============================================
echo   NEXT STEPS:
echo ============================================
echo.
echo   1. Collect data:    python collect_phrase.py
echo   2. Train model:     python train_phrase.py
echo   3. Run demo:        python demo_perfect.py
echo   4. Run web app:     python app.py
echo.
echo   Open http://localhost:5000 for the web platform
echo ============================================
pause
