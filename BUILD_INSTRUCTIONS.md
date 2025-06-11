# SpygateAI Desktop - Build Instructions

This document explains how to build SpygateAI Desktop into a professional installer for distribution.

## Prerequisites

### Required Software

1. **Python 3.8+** with PyInstaller

   ```bash
   pip install pyinstaller
   ```

2. **NSIS (Nullsoft Scriptable Install System)**
   - Download from: https://nsis.sourceforge.io/
   - Install to default location

### Optional (for advanced features)

- **UPX** (for smaller executable size)
- **Code signing certificate** (for trusted installer)

## Build Process

### Step 1: Build Executable

```bash
# Build the standalone executable
python build_installer.py
```

This creates:

- `dist/SpygateAI_Desktop.exe` - Standalone application
- `spygate_desktop.spec` - PyInstaller configuration
- `installer.nsi` - NSIS installer script
- `LICENSE.txt` - End user license agreement

### Step 2: Create Professional Installer

```bash
# Create the installer (requires NSIS)
makensis installer.nsi
```

This creates:

- `SpygateAI_Desktop_Installer.exe` - Professional installer

## Distribution Package

**What customers download:** `SpygateAI_Desktop_Installer.exe` (single file)

**What it includes:**

- ✅ Complete SpygateAI Desktop application
- ✅ All dependencies bundled (no Python required)
- ✅ Desktop and Start Menu shortcuts
- ✅ Professional uninstaller
- ✅ Windows Add/Remove Programs integration
- ✅ All assets (logo, formations, database)

## Testing the Build

### Test the Executable

```bash
# Test the standalone executable directly
./dist/SpygateAI_Desktop.exe
```

### Test the Installer

1. Run `SpygateAI_Desktop_Installer.exe`
2. Follow the installation wizard
3. Verify shortcuts are created
4. Launch from desktop shortcut
5. Test all application features
6. Test the uninstaller

## Troubleshooting

### Common Issues

**PyInstaller Import Errors:**

- Add missing modules to `hiddenimports` in the spec file
- Check console output for missing dependencies

**Large Executable Size:**

- Enable UPX compression in the spec file
- Remove unnecessary imports from the main application

**Missing Assets:**

- Verify all asset paths in the `datas` section of spec file
- Check that assets folder structure is preserved

**Installer Issues:**

- Ensure NSIS is installed and in PATH
- Check installer.nsi syntax with NSIS IDE

### File Size Optimization

The standalone executable may be large (~200-400MB) due to:

- PyQt6 framework
- OpenCV libraries
- NumPy dependencies
- PIL/Pillow imaging

This is normal for Python GUI applications with computer vision capabilities.

## Code Signing (Optional)

For trusted installers, sign both files:

```bash
# Sign the executable
signtool sign /f certificate.pfx /p password dist/SpygateAI_Desktop.exe

# Sign the installer
signtool sign /f certificate.pfx /p password SpygateAI_Desktop_Installer.exe
```

## Final Distribution

**For customers:**

- Distribute only: `SpygateAI_Desktop_Installer.exe`
- Include: `README_PURCHASE.md` (as text/email)
- Size: ~200-400MB (typical for modern desktop applications)

**Customer experience:**

1. Download single installer file
2. Run installer (may show Windows SmartScreen warning if unsigned)
3. Application installs to Program Files
4. Desktop shortcut created automatically
5. Launch and use immediately - no Python required!
