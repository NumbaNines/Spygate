#!/usr/bin/env python3
"""
Simple Security Fix for SpygateAI
Targeted approach to reduce security risk score
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime

class SimpleSecurityFix:
    def __init__(self):
        self.project_root = Path(".").resolve()
        self.fixes_applied = 0
        
    def remove_debug_files(self):
        """Remove common debug/temporary files that may contain sensitive data"""
        debug_patterns = [
            "*.log", "debug_*.py", "test_*.txt", 
            "temp_*.py", "backup_*.py"
        ]
        
        removed_files = []
        for pattern in debug_patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        removed_files.append(str(file_path))
                        self.fixes_applied += 1
                    except Exception as e:
                        print(f"Could not remove {file_path}: {e}")
        
        return removed_files

    def secure_config_files(self):
        """Fix common insecure configuration patterns"""
        config_files = ["production_config.py", "config.py"]
        fixed_files = []
        
        for config_file in config_files:
            config_paths = list(self.project_root.glob(f"**/{config_file}"))
            
            for config_path in config_paths:
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Fix DEBUG = True
                    content = re.sub(r'DEBUG\s*=\s*True', 'DEBUG = False', content, flags=re.IGNORECASE)
                    
                    # Fix empty ALLOWED_HOSTS
                    content = re.sub(r'ALLOWED_HOSTS\s*=\s*\[\s*\]', 'ALLOWED_HOSTS = ["localhost", "127.0.0.1"]', content)
                    
                    if content != original_content:
                        with open(config_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        fixed_files.append(str(config_path))
                        self.fixes_applied += 1
                        
                except Exception as e:
                    print(f"Could not fix {config_path}: {e}")
        
        return fixed_files

    def update_gitignore(self):
        """Enhance .gitignore with security patterns"""
        gitignore_path = self.project_root / ".gitignore"
        
        security_patterns = [
            "# Security",
            "*.key", "*.pem", "*.p12", "*.pfx",
            ".env", ".env.*", "!.env.example",
            "config.json", "secrets.json", "credentials.json",
            "*.log", "logs/",
            "# Database files",
            "*.db", "*.sqlite", "*.sqlite3",
            "# Temporary files", 
            "temp/", "tmp/", "cache/"
        ]
        
        try:
            if gitignore_path.exists():
                with open(gitignore_path, 'r') as f:
                    existing_content = f.read()
            else:
                existing_content = ""
            
            new_patterns = []
            for pattern in security_patterns:
                if pattern not in existing_content:
                    new_patterns.append(pattern)
            
            if new_patterns:
                with open(gitignore_path, 'a') as f:
                    f.write("\n" + "\n".join(new_patterns) + "\n")
                self.fixes_applied += 1
                return True
                
        except Exception as e:
            print(f"Could not update .gitignore: {e}")
        
        return False

    def clean_sensitive_keywords(self):
        """Remove files with obviously sensitive content"""
        sensitive_files = []
        
        # Look for files with sensitive names
        sensitive_patterns = [
            "*password*", "*secret*", "*key*", "*token*",
            "*credential*", "*auth*"
        ]
        
        for pattern in sensitive_patterns:
            for file_path in self.project_root.glob(f"**/{pattern}"):
                if (file_path.is_file() and 
                    file_path.suffix in {'.txt', '.json', '.cfg', '.conf'} and
                    'test' not in str(file_path).lower() and
                    'example' not in str(file_path).lower()):
                    
                    try:
                        # Move to security folder instead of deleting
                        security_dir = self.project_root / "security" / "quarantine"
                        security_dir.mkdir(parents=True, exist_ok=True)
                        
                        new_path = security_dir / file_path.name
                        shutil.move(str(file_path), str(new_path))
                        sensitive_files.append(str(file_path))
                        self.fixes_applied += 1
                        
                    except Exception as e:
                        print(f"Could not quarantine {file_path}: {e}")
        
        return sensitive_files

    def run_fixes(self):
        """Run all security fixes"""
        print("🔒 Simple Security Fix - Step 11")
        print("=" * 50)
        print("🔧 Applying targeted security fixes...")
        print()
        
        # 1. Remove debug files
        print("1️⃣  Removing debug/temp files...")
        removed = self.remove_debug_files()
        if removed:
            print(f"   ✅ Removed {len(removed)} debug files")
        
        # 2. Fix config files
        print("2️⃣  Securing configuration files...")
        fixed_configs = self.secure_config_files()
        if fixed_configs:
            print(f"   ✅ Fixed {len(fixed_configs)} config files")
        
        # 3. Update .gitignore
        print("3️⃣  Updating .gitignore...")
        if self.update_gitignore():
            print("   ✅ Enhanced .gitignore with security patterns")
        
        # 4. Quarantine sensitive files
        print("4️⃣  Quarantining sensitive files...")
        quarantined = self.clean_sensitive_keywords()
        if quarantined:
            print(f"   ✅ Quarantined {len(quarantined)} sensitive files")
        
        print()
        print("=" * 50)
        print(f"🎯 SECURITY FIXES COMPLETE")
        print(f"🔧 Total fixes applied: {self.fixes_applied}")
        print("=" * 50)

if __name__ == "__main__":
    fixer = SimpleSecurityFix()
    fixer.run_fixes() 