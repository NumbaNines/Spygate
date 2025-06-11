#!/usr/bin/env powershell

Write-Host "🔥 Starting SpygateAI with GPU + OCR Support" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "📍 Activating GPU environment..." -ForegroundColor Yellow
conda activate spygate-gpu

Write-Host "✅ Verifying dependencies..." -ForegroundColor Green
python -c "import torch; print('✅ PyTorch + CUDA:', torch.cuda.is_available())"
python -c "import pytesseract; print('✅ PyTesseract ready')"
python -c "from ultralytics import YOLO; print('✅ YOLOv8 ready')"

Write-Host ""
Write-Host "🚀 Launching SpygateAI Desktop App..." -ForegroundColor Green
Write-Host ""
python spygate_desktop_app_faceit_style.py 