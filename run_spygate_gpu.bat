@echo off
echo 🔥 Starting SpygateAI with GPU + OCR Support
echo ===============================================
echo.

echo 📍 Activating GPU environment...
call conda activate spygate-gpu

echo ✅ Verifying dependencies...
python -c "import torch; print('✅ PyTorch + CUDA:', torch.cuda.is_available())"
python -c "import pytesseract; print('✅ PyTesseract ready')"
python -c "from ultralytics import YOLO; print('✅ YOLOv8 ready')"

echo.
echo 🚀 Launching SpygateAI Desktop App...
echo.
python spygate_desktop_app_faceit_style.py

pause 