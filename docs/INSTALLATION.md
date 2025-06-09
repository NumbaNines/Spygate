## Installing Tesseract OCR

SpygateAI uses Tesseract OCR for extracting text from game HUD elements. Follow these instructions to install Tesseract on your system:

### Windows

1. Download the latest Tesseract installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer and make note of the installation path (default is `C:\Program Files\Tesseract-OCR`)
3. Add the Tesseract installation directory to your system's PATH environment variable
4. Verify the installation by opening a new terminal and running:
   ```
   tesseract --version
   ```

### macOS

Using Homebrew:

```bash
brew install tesseract
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install libtesseract-dev
```

### Verifying the Installation

After installing Tesseract, you can verify that SpygateAI can find it by running:

```python
import pytesseract
print(pytesseract.get_tesseract_version())
```

If you get an error about Tesseract not being found, you may need to explicitly set the path in your code:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example
```
