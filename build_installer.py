#!/usr/bin/env python3
"""
SpygateAI Desktop - Build Script
================================

This script builds the SpygateAI Desktop application into:
1. A standalone executable (no Python required)
2. A professional Windows installer (.exe)

Requirements:
    pip install pyinstaller
    
Usage:
    python build_installer.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def check_build_requirements():
    """Check if build requirements are installed"""
    try:
        import PyInstaller
        print("✅ PyInstaller found")
    except ImportError:
        print("❌ PyInstaller not found. Install with: pip install pyinstaller")
        return False
    
    return True

def clean_build_dirs():
    """Clean previous build directories"""
    dirs_to_clean = ["build", "dist", "__pycache__"]
    
    for dir_name in dirs_to_clean:
        if Path(dir_name).exists():
            print(f"🧹 Cleaning {dir_name}/")
            shutil.rmtree(dir_name)

def create_pyinstaller_spec():
    """Create PyInstaller spec file for SpygateAI Desktop"""
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['spygate_desktop.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('assets', 'assets'),
        ('spygate_users.db', '.'),
        ('DESKTOP_README.md', '.'),
        ('README_PURCHASE.md', '.'),
    ],
    hiddenimports=[
        'PyQt6.QtCore',
        'PyQt6.QtGui', 
        'PyQt6.QtWidgets',
        'cv2',
        'numpy',
        'PIL',
        'sqlite3',
        'json',
        'pathlib',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SpygateAI_Desktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon if available
    version_file=None  # Add version info if available
)
'''
    
    with open("spygate_desktop.spec", "w") as f:
        f.write(spec_content)
    
    print("✅ Created PyInstaller spec file")

def build_executable():
    """Build the standalone executable"""
    print("🔨 Building SpygateAI Desktop executable...")
    
    try:
        # Run PyInstaller
        result = subprocess.run([
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm", 
            "spygate_desktop.spec"
        ], check=True, capture_output=True, text=True)
        
        print("✅ Executable built successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_installer_script():
    """Create NSIS installer script"""
    nsis_script = '''
; SpygateAI Desktop Installer Script
; Built with NSIS (Nullsoft Scriptable Install System)

!define APP_NAME "SpygateAI Desktop"
!define APP_VERSION "1.0"
!define APP_PUBLISHER "SpygateAI"
!define APP_URL "https://spygateai.com"
!define APP_EXECUTABLE "SpygateAI_Desktop.exe"

; Modern UI
!include "MUI2.nsh"

; General settings
Name "${APP_NAME}"
OutFile "SpygateAI_Desktop_Installer.exe"
Unicode True
InstallDir "$PROGRAMFILES64\\${APP_NAME}"
InstallDirRegKey HKCU "Software\\${APP_NAME}" ""
RequestExecutionLevel admin

; Modern UI Configuration
!define MUI_ABORTWARNING
!define MUI_ICON "assets\\logo\\spygate-logo.ico"
!define MUI_WELCOMEPAGE_TITLE "${APP_NAME} Setup"
!define MUI_WELCOMEPAGE_TEXT "Welcome to the ${APP_NAME} Setup Wizard.$\\r$\\n$\\r$\\nThis will install ${APP_NAME} on your computer."

; Pages
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE.txt"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_WELCOME
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

; Languages
!insertmacro MUI_LANGUAGE "English"

; Installer Section
Section "Install"
    SetOutPath "$INSTDIR"
    
    ; Copy main files
    File "dist\\${APP_EXECUTABLE}"
    File /r "dist\\*.*"
    
    ; Create shortcuts
    CreateDirectory "$SMPROGRAMS\\${APP_NAME}"
    CreateShortcut "$SMPROGRAMS\\${APP_NAME}\\${APP_NAME}.lnk" "$INSTDIR\\${APP_EXECUTABLE}"
    CreateShortcut "$DESKTOP\\${APP_NAME}.lnk" "$INSTDIR\\${APP_EXECUTABLE}"
    
    ; Registry entries
    WriteRegStr HKCU "Software\\${APP_NAME}" "" $INSTDIR
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
    
    ; Add to Add/Remove Programs
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "DisplayName" "${APP_NAME}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "UninstallString" "$INSTDIR\\Uninstall.exe"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "Publisher" "${APP_PUBLISHER}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "URLInfoAbout" "${APP_URL}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}" "DisplayVersion" "${APP_VERSION}"
SectionEnd

; Uninstaller Section
Section "Uninstall"
    Delete "$INSTDIR\\${APP_EXECUTABLE}"
    Delete "$INSTDIR\\Uninstall.exe"
    RMDir /r "$INSTDIR"
    
    Delete "$SMPROGRAMS\\${APP_NAME}\\${APP_NAME}.lnk"
    RMDir "$SMPROGRAMS\\${APP_NAME}"
    Delete "$DESKTOP\\${APP_NAME}.lnk"
    
    DeleteRegKey HKCU "Software\\${APP_NAME}"
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\${APP_NAME}"
SectionEnd
'''
    
    with open("installer.nsi", "w") as f:
        f.write(nsis_script)
    
    print("✅ Created NSIS installer script")

def create_simple_license():
    """Create a simple license file for the installer"""
    license_text = """SpygateAI Desktop - End User License Agreement

Copyright (c) 2024 SpygateAI. All rights reserved.

This software is licensed, not sold. By installing and using SpygateAI Desktop, you agree to the following terms:

1. LICENSE GRANT
   Subject to the terms of this agreement, SpygateAI grants you a non-exclusive license to use SpygateAI Desktop.

2. RESTRICTIONS
   You may not distribute, modify, reverse engineer, or create derivative works of this software.

3. PRIVACY
   SpygateAI respects your privacy. User data is stored locally on your device.

4. SUPPORT
   Technical support is available at support@spygateai.com

5. WARRANTIES
   This software is provided "as is" without warranty of any kind.

By installing this software, you acknowledge that you have read and agree to these terms.
"""
    
    with open("LICENSE.txt", "w") as f:
        f.write(license_text)
    
    print("✅ Created license file")

def main():
    """Main build process"""
    print("🏈 SpygateAI Desktop - Build Process")
    print("=" * 40)
    
    # Check requirements
    if not check_build_requirements():
        sys.exit(1)
    
    # Clean previous builds
    clean_build_dirs()
    
    # Create build files
    create_pyinstaller_spec()
    create_simple_license()
    
    # Build executable
    if not build_executable():
        sys.exit(1)
    
    # Create installer script
    create_installer_script()
    
    print("\n🎉 Build Process Complete!")
    print("\nNext steps:")
    print("1. ✅ Executable created: dist/SpygateAI_Desktop.exe")
    print("2. 📝 To create installer, install NSIS and run:")
    print("   makensis installer.nsi")
    print("3. 🚀 Distribute: SpygateAI_Desktop_Installer.exe")
    
    print("\n📦 Professional Distribution Package:")
    print("   • Single installer file (.exe)")
    print("   • No Python installation required")
    print("   • Desktop & Start Menu shortcuts")
    print("   • Proper uninstaller")
    print("   • Windows Add/Remove Programs integration")

if __name__ == "__main__":
    main() 