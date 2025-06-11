
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
InstallDir "$PROGRAMFILES64\${APP_NAME}"
InstallDirRegKey HKCU "Software\${APP_NAME}" ""
RequestExecutionLevel admin

; Modern UI Configuration
!define MUI_ABORTWARNING
!define MUI_ICON "assets\logo\spygate-logo.ico"
!define MUI_WELCOMEPAGE_TITLE "${APP_NAME} Setup"
!define MUI_WELCOMEPAGE_TEXT "Welcome to the ${APP_NAME} Setup Wizard.$\r$\n$\r$\nThis will install ${APP_NAME} on your computer."

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
    File "dist\${APP_EXECUTABLE}"
    File /r "dist\*.*"
    
    ; Create shortcuts
    CreateDirectory "$SMPROGRAMS\${APP_NAME}"
    CreateShortcut "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk" "$INSTDIR\${APP_EXECUTABLE}"
    CreateShortcut "$DESKTOP\${APP_NAME}.lnk" "$INSTDIR\${APP_EXECUTABLE}"
    
    ; Registry entries
    WriteRegStr HKCU "Software\${APP_NAME}" "" $INSTDIR
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    
    ; Add to Add/Remove Programs
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayName" "${APP_NAME}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "Publisher" "${APP_PUBLISHER}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "URLInfoAbout" "${APP_URL}"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}" "DisplayVersion" "${APP_VERSION}"
SectionEnd

; Uninstaller Section
Section "Uninstall"
    Delete "$INSTDIR\${APP_EXECUTABLE}"
    Delete "$INSTDIR\Uninstall.exe"
    RMDir /r "$INSTDIR"
    
    Delete "$SMPROGRAMS\${APP_NAME}\${APP_NAME}.lnk"
    RMDir "$SMPROGRAMS\${APP_NAME}"
    Delete "$DESKTOP\${APP_NAME}.lnk"
    
    DeleteRegKey HKCU "Software\${APP_NAME}"
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\${APP_NAME}"
SectionEnd
