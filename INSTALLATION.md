# SpygateAI Installation Guide

## 🚀 Quick Start (Recommended)

### Option 1: Use the GPU Launcher Scripts

1. **Double-click**: `run_spygate_gpu.bat`
2. **Or run**: `.\run_spygate_gpu.ps1` in PowerShell

The launcher scripts will automatically:

- ✅ Activate the GPU environment
- ✅ Verify all dependencies
- ✅ Start SpygateAI with full GPU + OCR support

---

## 📋 Manual Installation

### Prerequisites

- **Windows 10/11** 64-bit
- **Python 3.11+** (Python 3.12 recommended)
- **NVIDIA GPU** with CUDA support (RTX series recommended)
- **16GB+ RAM** (32GB for large video files)

### Step 1: Install Miniconda

1. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Choose Windows 64-bit installer
3. Install with default settings

### Step 2: Create GPU Environment

```bash
# Open Anaconda Prompt
conda create -n spygate-gpu python=3.11 -y
conda activate spygate-gpu
```

### Step 3: Install PyTorch with CUDA

```bash
# For CUDA 12.1 (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify GPU support
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Step 4: Install SpygateAI Dependencies

```bash
# Navigate to project directory
cd /path/to/SpygateAI

# Install all dependencies
pip install -r requirements.txt
```

### Step 5: Install Tesseract OCR

```bash
# Install via winget
winget install UB-Mannheim.TesseractOCR

# Or download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Step 6: Verify Installation

```bash
# Test complete setup
python -c "
import torch, ultralytics, pytesseract
print('✅ PyTorch + CUDA:', torch.cuda.is_available())
print('✅ YOLOv8:', ultralytics.__version__)
print('✅ OCR ready')
"
```

---

## 🔧 Environment Configuration

### Setting Python Interpreter in IDEs

#### Visual Studio Code / Cursor

1. Press `Ctrl + Shift + P`
2. Type "Python: Select Interpreter"
3. Choose: `C:\Users\{USER}\miniconda3\envs\spygate-gpu\python.exe`

#### PyCharm

1. File → Settings → Project → Python Interpreter
2. Add New → Conda Environment → Existing
3. Select: `C:\Users\{USER}\miniconda3\envs\spygate-gpu\python.exe`

---

## ⚡ Performance Optimization

### GPU Memory Optimization

```python
# Add to your scripts for optimal GPU usage
import torch
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.cuda.empty_cache()  # Clear GPU cache
```

### For RTX 40-Series GPUs

```bash
# Enhanced performance with latest CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🐛 Troubleshooting

### Common Issues

#### PyTorch CUDA Not Available

```bash
# Reinstall with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### PyTesseract Module Not Found

```bash
# Ensure you're in the correct environment
conda activate spygate-gpu
pip install pytesseract
```

#### Environment Not Found

```bash
# Recreate the environment
conda remove -n spygate-gpu --all -y
conda create -n spygate-gpu python=3.11 -y
# Then repeat Steps 3-4
```

#### GUI Application Won't Start

```bash
# Install PyQt6 in the correct environment
conda activate spygate-gpu
pip install PyQt6
```

---

## 📊 System Requirements

### Minimum Requirements

- **CPU**: Intel i5-8400 / AMD Ryzen 5 2600
- **GPU**: GTX 1060 6GB / RTX 2060
- **RAM**: 16GB
- **Storage**: 10GB free space

### Recommended Requirements

- **CPU**: Intel i7-10700K / AMD Ryzen 7 3700X
- **GPU**: RTX 3070 / RTX 4060 or higher
- **RAM**: 32GB
- **Storage**: 50GB free space (SSD recommended)

### Optimal Performance

- **CPU**: Intel i9-12900K / AMD Ryzen 9 5900X
- **GPU**: RTX 4070 SUPER / RTX 4080 or higher
- **RAM**: 64GB DDR4-3200+
- **Storage**: 100GB+ NVMe SSD

---

## 🚀 Performance Expectations

| Hardware       | Video Analysis Speed | Model Loading |
| -------------- | -------------------- | ------------- |
| RTX 4070 SUPER | 60-200 FPS           | 1-3 seconds   |
| RTX 3070       | 40-120 FPS           | 2-4 seconds   |
| RTX 2070       | 25-80 FPS            | 3-6 seconds   |
| CPU Only       | 2-10 FPS             | 10-30 seconds |

---

## 📞 Support

If you encounter issues:

1. Check this troubleshooting guide
2. Verify your environment with the launcher scripts
3. Ensure all dependencies are in the `spygate-gpu` environment
4. Check GPU drivers are up to date

---

## 🔄 Updates

To update SpygateAI:

```bash
conda activate spygate-gpu
git pull
pip install -r requirements.txt --upgrade
```
