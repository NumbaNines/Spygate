#!/usr/bin/env python3
"""
SpygateAI Security Hardening System - Step 11
Implements security improvements and hardens the application
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


class SecurityHardener:
    """Security hardening implementation for SpygateAI"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.security_dir = self.project_root / "security"
        self.backup_dir = self.security_dir / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        print("🛡️  SpygateAI Security Hardener Initialized")
        print(f"📁 Project Root: {self.project_root.absolute()}")
        
    def implement_hardening(self, audit_file: str = None) -> Dict:
        """Implement security hardening measures"""
        print("\n🔧 Starting Security Hardening...")
        print("=" * 60)
        
        hardening_results = {
            "timestamp": datetime.now().isoformat(),
            "hardening_applied": [],
            "files_modified": [],
            "backup_created": None,
            "status": "in_progress"
        }
        
        try:
            # 1. Create backup
            print("\n1️⃣  Creating Security Backup...")
            backup_path = self._create_security_backup()
            hardening_results["backup_created"] = str(backup_path)
            
            # 2. Implement Git security
            print("\n2️⃣  Implementing Git Security...")
            git_changes = self._implement_git_security()
            hardening_results["hardening_applied"].extend(git_changes)
            
            # 3. Secure file permissions
            print("\n3️⃣  Securing File Permissions...")
            perm_changes = self._secure_file_permissions()
            hardening_results["hardening_applied"].extend(perm_changes)
            
            # 4. Create security configuration
            print("\n4️⃣  Creating Security Configuration...")
            config_changes = self._create_security_config()
            hardening_results["hardening_applied"].extend(config_changes)
            
            # 5. Implement dependency security
            print("\n5️⃣  Implementing Dependency Security...")
            dep_changes = self._implement_dependency_security()
            hardening_results["hardening_applied"].extend(dep_changes)
            
            # 6. Create security documentation
            print("\n6️⃣  Creating Security Documentation...")
            doc_changes = self._create_security_documentation()
            hardening_results["hardening_applied"].extend(doc_changes)
            
            hardening_results["status"] = "completed"
            
        except Exception as e:
            print(f"❌ Hardening failed: {e}")
            hardening_results["status"] = "failed"
            hardening_results["error"] = str(e)
        
        return hardening_results
    
    def _create_security_backup(self) -> Path:
        """Create backup of critical files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"security_backup_{timestamp}"
        backup_path.mkdir(exist_ok=True)
        
        # Backup critical files
        critical_files = ["requirements.txt", ".gitignore"]
        backed_up = 0
        
        for file_name in critical_files:
            file_path = self.project_root / file_name
            if file_path.exists():
                try:
                    shutil.copy2(file_path, backup_path / file_name)
                    backed_up += 1
                except Exception:
                    continue
        
        print(f"   💾 Backed up {backed_up} critical files to {backup_path.name}")
        return backup_path
    
    def _implement_git_security(self) -> List[str]:
        """Implement Git security measures"""
        changes = []
        
        # Create/update .gitignore
        gitignore_path = self.project_root / ".gitignore"
        security_entries = [
            "# Security sensitive files",
            "*.key", "*.pem", "*.p12", "*.pfx",
            ".env", ".env.local", ".env.production",
            "secrets.json", "config/secrets.yml",
            "", "# Database files", "*.db", "*.sqlite3",
            "", "# Log files", "*.log", "logs/",
            "", "# Temporary files", "temp/", "tmp/", "*.tmp"
        ]
        
        try:
            existing_content = ""
            if gitignore_path.exists():
                existing_content = gitignore_path.read_text(encoding='utf-8')
            
            new_entries = []
            for entry in security_entries:
                if entry and entry not in existing_content:
                    new_entries.append(entry)
            
            if new_entries:
                with open(gitignore_path, 'a', encoding='utf-8') as f:
                    f.write('\n\n' + '\n'.join(new_entries))
                changes.append("Updated .gitignore with security patterns")
                print(f"   📝 Added {len(new_entries)} security entries to .gitignore")
            else:
                print("   ✅ .gitignore already has security entries")
                
        except Exception as e:
            print(f"   ❌ Failed to update .gitignore: {e}")
        
        return changes
    
    def _secure_file_permissions(self) -> List[str]:
        """Secure file permissions"""
        changes = []
        
        try:
            # Secure Python files (limit for performance)
            python_files = list(self.project_root.rglob("*.py"))
            secured_count = 0
            
            for py_file in python_files[:100]:  # Limit for performance
                try:
                    current_mode = py_file.stat().st_mode
                    new_mode = current_mode & ~0o077  # Remove world permissions
                    py_file.chmod(new_mode)
                    secured_count += 1
                except (OSError, PermissionError):
                    continue
            
            if secured_count > 0:
                changes.append(f"Secured permissions for {secured_count} Python files")
                print(f"   🔒 Secured {secured_count} Python files")
            
        except Exception as e:
            print(f"   ❌ Failed to secure file permissions: {e}")
        
        return changes
    
    def _create_security_config(self) -> List[str]:
        """Create security configuration files"""
        changes = []
        
        # Create security policy file
        security_policy = {
            "security_policy": {
                "version": "1.0",
                "created": datetime.now().isoformat(),
                "requirements": {
                    "min_password_length": 12,
                    "require_2fa": True,
                    "session_timeout_minutes": 30,
                    "max_login_attempts": 3
                },
                "file_permissions": {
                    "python_files": "0o640",
                    "config_files": "0o640", 
                    "log_files": "0o600",
                    "database_files": "0o600"
                }
            }
        }
        
        try:
            policy_file = self.security_dir / "security_policy.json"
            with open(policy_file, 'w', encoding='utf-8') as f:
                json.dump(security_policy, f, indent=2)
            
            changes.append("Created security policy configuration")
            print(f"   📋 Created security policy: {policy_file.name}")
            
        except Exception as e:
            print(f"   ❌ Failed to create security policy: {e}")
        
        # Create environment template
        env_template = """# SpygateAI Environment Configuration
# Copy this file to .env and fill in your values

# API Keys (keep these secret!)
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here

# Security Configuration
SECRET_KEY=your_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here

# Application Settings
DEBUG=False
ENVIRONMENT=production
LOG_LEVEL=INFO
"""
        
        try:
            env_file = self.security_dir / "env.template"
            with open(env_file, 'w', encoding='utf-8') as f:
                f.write(env_template)
            
            changes.append("Created environment variable template")
            print(f"   🌍 Created environment template: {env_file.name}")
            
        except Exception as e:
            print(f"   ❌ Failed to create environment template: {e}")
        
        return changes
    
    def _implement_dependency_security(self) -> List[str]:
        """Implement dependency security measures"""
        changes = []
        
        # Create requirements-security.txt for security tools
        security_tools = [
            "bandit>=1.7.0",
            "safety>=2.0.0", 
            "pip-audit>=2.0.0"
        ]
        
        try:
            security_req_file = self.project_root / "requirements-security.txt"
            with open(security_req_file, 'w', encoding='utf-8') as f:
                f.write("# Security analysis tools\n")
                f.write("# Install with: pip install -r requirements-security.txt\n\n")
                f.write('\n'.join(security_tools))
            
            changes.append("Created security tools requirements file")
            print(f"   🔧 Created requirements-security.txt")
            
        except Exception as e:
            print(f"   ❌ Failed to create security requirements: {e}")
        
        return changes
    
    def _create_security_documentation(self) -> List[str]:
        """Create security documentation"""
        changes = []
        
        # Create security README
        security_readme = f"""# SpygateAI Security Documentation

## Overview
This document outlines the security measures implemented for SpygateAI.

## Security Hardening Applied
Generated on: {datetime.now().isoformat()}

### File Security
- Secured file permissions for Python files
- Updated .gitignore with security patterns
- Created backup of critical files

### Configuration Security
- Created security policy configuration
- Provided environment variable template
- Implemented secure defaults

### Git Security
- Added comprehensive .gitignore patterns
- Protected sensitive files from version control

## Security Policy
See `security_policy.json` for detailed security requirements.

## Environment Configuration
1. Copy `env.template` to `.env`
2. Fill in your actual values (never commit .env files)
3. Ensure proper file permissions: `chmod 600 .env`

## Security Tools
Install security analysis tools:
```bash
pip install -r requirements-security.txt
```

## Regular Security Tasks
- [ ] Run security audits monthly
- [ ] Update dependencies quarterly  
- [ ] Review access permissions quarterly
"""
        
        try:
            readme_file = self.security_dir / "README.md"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(security_readme)
            
            changes.append("Created security documentation")
            print(f"   📚 Created security README: {readme_file.name}")
            
        except Exception as e:
            print(f"   ❌ Failed to create security documentation: {e}")
        
        return changes
    
    def save_hardening_report(self, hardening_results: Dict):
        """Save hardening report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.security_dir / f"security_hardening_{timestamp}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(hardening_results, f, indent=2, default=str)
            
            print(f"\n📋 Security Hardening Report Saved:")
            print(f"   📄 File: {report_file.name}")
            print(f"   🔧 Measures Applied: {len(hardening_results['hardening_applied'])}")
            print(f"   📁 Status: {hardening_results['status']}")
            
            return report_file
            
        except Exception as e:
            print(f"❌ Failed to save hardening report: {e}")
            return None


def main():
    """Main function"""
    print("🛡️  SpygateAI Security Hardening System - Step 11")
    print("=" * 50)
    
    hardener = SecurityHardener()
    hardening_results = hardener.implement_hardening()
    report_file = hardener.save_hardening_report(hardening_results)
    
    print("\n" + "=" * 60)
    print("🎯 SECURITY HARDENING COMPLETE")
    print("=" * 60)
    
    print(f"🔧 Status: {hardening_results['status'].upper()}")
    print(f"🛡️  Measures Applied: {len(hardening_results['hardening_applied'])}")
    
    if hardening_results["hardening_applied"]:
        print(f"\n📝 Security Measures Applied:")
        for i, measure in enumerate(hardening_results["hardening_applied"], 1):
            print(f"   {i}. {measure}")
    
    return hardening_results


if __name__ == "__main__":
    main()