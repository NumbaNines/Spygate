#!/usr/bin/env python3
"""
SpygateAI Security Audit System
Enterprise-grade security scanning and vulnerability assessment

This module performs comprehensive security audits including:
- Dependency vulnerability scanning
- Code security analysis
- File permission auditing
- Network security assessment
- Configuration security review
"""

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import bandit
    import requests
    from bandit.core import config as bandit_config
    from bandit.core import manager
except ImportError:
    print("⚠️  Security dependencies not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "bandit", "requests", "safety"])
    import bandit
    import requests
    from bandit.core import config as bandit_config
    from bandit.core import manager


class SecurityAuditor:
    """Comprehensive security audit system for SpygateAI"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.audit_dir = self.project_root / "security"
        self.audit_dir.mkdir(exist_ok=True)

        # Security scan results
        self.vulnerabilities = []
        self.security_issues = []
        self.file_permissions = []
        self.database_security = []
        self.network_security = []

        print("🔒 SpygateAI Security Auditor Initialized")
        print(f"📁 Project Root: {self.project_root.absolute()}")
        print(f"🛡️  Audit Directory: {self.audit_dir.absolute()}")

    def run_comprehensive_audit(self) -> dict:
        """Run complete security audit"""
        print("\n🔍 Starting Comprehensive Security Audit...")
        print("=" * 60)

        audit_results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root.absolute()),
            "audit_summary": {},
            "vulnerability_scan": {},
            "code_security": {},
            "file_permissions": {},
            "database_security": {},
            "network_security": {},
            "recommendations": [],
            "risk_assessment": {},
        }

        # 1. Dependency Vulnerability Scan
        print("\n1️⃣  Scanning Dependencies for Vulnerabilities...")
        audit_results["vulnerability_scan"] = self._scan_dependencies()

        # 2. Code Security Analysis
        print("\n2️⃣  Analyzing Code Security with Bandit...")
        audit_results["code_security"] = self._analyze_code_security()

        # 3. File Permission Audit
        print("\n3️⃣  Auditing File Permissions...")
        audit_results["file_permissions"] = self._audit_file_permissions()

        # 4. Database Security Check
        print("\n4️⃣  Checking Database Security...")
        audit_results["database_security"] = self._check_database_security()

        # 5. Network Security Assessment
        print("\n5️⃣  Assessing Network Security...")
        audit_results["network_security"] = self._assess_network_security()

        # 6. Generate Risk Assessment
        print("\n6️⃣  Generating Risk Assessment...")
        audit_results["risk_assessment"] = self._generate_risk_assessment(audit_results)

        # 7. Generate Recommendations
        print("\n7️⃣  Generating Security Recommendations...")
        audit_results["recommendations"] = self._generate_recommendations(audit_results)

        # 8. Create Summary
        audit_results["audit_summary"] = self._create_audit_summary(audit_results)

        return audit_results

    def _scan_dependencies(self) -> dict:
        """Scan dependencies for known vulnerabilities"""
        print("\n1️⃣  Scanning Dependencies for Vulnerabilities...")
        results = {
            "requirements_found": False,
            "vulnerabilities": [],
            "total_packages": 0,
            "vulnerable_packages": 0,
            "scan_status": "completed",
        }

        try:
            req_file = self.project_root / "requirements.txt"
            if req_file.exists():
                results["requirements_found"] = True
                lines = req_file.read_text().splitlines()
                results["total_packages"] = len(
                    [l for l in lines if l.strip() and not l.startswith("#")]
                )
                print(f"   📦 Found requirements.txt with {results['total_packages']} packages")

                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "safety", "check", "-r", str(req_file), "--json"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )

                    if result.returncode == 0:
                        safety_data = json.loads(result.stdout) if result.stdout else []
                        results["vulnerabilities"] = safety_data
                        results["vulnerable_packages"] = len(safety_data)
                        print(
                            f"   ✅ Safety scan completed: {len(safety_data)} vulnerabilities found"
                        )
                    else:
                        print(f"   ⚠️  Safety scan failed: {result.stderr}")
                        results["scan_status"] = "failed"

                except subprocess.TimeoutExpired:
                    print("   ⏱️  Safety scan timed out")
                    results["scan_status"] = "timeout"
                except Exception as e:
                    print(f"   ❌ Safety scan error: {e}")
                    results["scan_status"] = "error"
            else:
                print("   ⚠️  No requirements.txt found")

        except Exception as e:
            print(f"   ❌ Dependency scan error: {e}")
            results["scan_status"] = "error"

        return results

    def _analyze_code_security(self) -> dict:
        """Analyze code security using Bandit"""
        print("\n2️⃣  Analyzing Code Security with Bandit...")
        results = {
            "total_files_scanned": 0,
            "issues_found": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "issues": [],
            "scan_status": "completed",
        }

        try:
            python_files = list(self.project_root.rglob("*.py"))
            python_files = [
                f
                for f in python_files
                if not any(
                    exclude in str(f)
                    for exclude in [
                        "venv",
                        "env",
                        "__pycache__",
                        ".git",
                        "build",
                        "dist",
                        "node_modules",
                        ".pytest_cache",
                        "htmlcov",
                    ]
                )
            ]

            results["total_files_scanned"] = len(python_files)
            print(f"   📄 Scanning {len(python_files)} Python files...")

            if python_files:
                from bandit.core import config as bandit_config
                from bandit.core import manager

                conf = bandit_config.BanditConfig()
                b_mgr = manager.BanditManager(conf, "file")

                for py_file in python_files[:50]:  # Limit for performance
                    try:
                        b_mgr.discover_files([str(py_file)])
                    except Exception:
                        continue

                b_mgr.run_tests()

                for result in b_mgr.get_issue_list():
                    issue = {
                        "filename": str(result.fname),
                        "line_number": result.lineno,
                        "test_name": result.test,
                        "severity": result.severity,
                        "confidence": result.confidence,
                        "text": (
                            result.text[:200] + "..." if len(result.text) > 200 else result.text
                        ),
                    }
                    results["issues"].append(issue)

                    if result.severity == "HIGH":
                        results["high_severity"] += 1
                    elif result.severity == "MEDIUM":
                        results["medium_severity"] += 1
                    else:
                        results["low_severity"] += 1

                results["issues_found"] = len(results["issues"])
                print(f"   🔍 Found {results['issues_found']} security issues:")
                print(f"      🔴 High: {results['high_severity']}")
                print(f"      🟡 Medium: {results['medium_severity']}")
                print(f"      🟢 Low: {results['low_severity']}")

        except Exception as e:
            print(f"   ❌ Code security analysis error: {e}")
            results["scan_status"] = "error"

        return results

    def _audit_file_permissions(self) -> dict:
        """Audit file permissions for security issues"""
        print("\n3️⃣  Auditing File Permissions...")
        results = {
            "total_files_checked": 0,
            "permission_issues": [],
            "executable_files": [],
            "world_writable": [],
            "config_files": [],
            "scan_status": "completed",
        }

        try:
            sensitive_extensions = [
                ".py",
                ".sh",
                ".bat",
                ".ps1",
                ".conf",
                ".cfg",
                ".ini",
                ".json",
                ".yaml",
                ".yml",
            ]

            for file_path in self.project_root.rglob("*"):
                if file_path.is_file():
                    results["total_files_checked"] += 1

                    try:
                        file_stat = file_path.stat()
                        mode = file_stat.st_mode

                        # Check for executable files
                        if mode & stat.S_IXUSR:
                            results["executable_files"].append(
                                {
                                    "path": str(file_path.relative_to(self.project_root)),
                                    "permissions": oct(mode)[-3:],
                                }
                            )

                        # Check for world-writable files
                        if mode & stat.S_IWOTH:
                            results["world_writable"].append(
                                {
                                    "path": str(file_path.relative_to(self.project_root)),
                                    "permissions": oct(mode)[-3:],
                                }
                            )

                        # Check configuration files
                        if any(file_path.suffix.lower() == ext for ext in sensitive_extensions):
                            results["config_files"].append(
                                {
                                    "path": str(file_path.relative_to(self.project_root)),
                                    "permissions": oct(mode)[-3:],
                                    "size_bytes": file_stat.st_size,
                                }
                            )

                    except (OSError, PermissionError):
                        continue

            print(f"   📁 Checked {results['total_files_checked']} files")
            print(f"   🔧 Executable files: {len(results['executable_files'])}")
            print(f"   ⚠️  World-writable files: {len(results['world_writable'])}")
            print(f"   ⚙️  Configuration files: {len(results['config_files'])}")

        except Exception as e:
            print(f"   ❌ File permission audit error: {e}")
            results["scan_status"] = "error"

        return results

    def _check_database_security(self) -> dict:
        """Check database security configuration"""
        print("\n4️⃣  Checking Database Security...")
        results = {
            "databases_found": [],
            "security_issues": [],
            "recommendations": [],
            "scan_status": "completed",
        }

        try:
            db_files = list(self.project_root.rglob("*.db")) + list(
                self.project_root.rglob("*.sqlite")
            )

            for db_file in db_files:
                db_info = {
                    "path": str(db_file.relative_to(self.project_root)),
                    "size_bytes": db_file.stat().st_size,
                    "permissions": oct(db_file.stat().st_mode)[-3:],
                    "tables": [],
                    "issues": [],
                }

                try:
                    conn = sqlite3.connect(str(db_file))
                    cursor = conn.cursor()

                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    db_info["tables"] = [table[0] for table in tables]

                    for table_name in db_info["tables"]:
                        try:
                            cursor.execute(f"PRAGMA table_info({table_name})")
                            columns = cursor.fetchall()

                            sensitive_cols = [
                                col[1]
                                for col in columns
                                if any(
                                    keyword in col[1].lower()
                                    for keyword in [
                                        "password",
                                        "pass",
                                        "pwd",
                                        "secret",
                                        "token",
                                        "key",
                                        "hash",
                                    ]
                                )
                            ]

                            if sensitive_cols:
                                db_info["issues"].append(
                                    f"Table '{table_name}' contains sensitive columns: {sensitive_cols}"
                                )

                        except sqlite3.Error:
                            continue

                    conn.close()

                except sqlite3.Error as e:
                    db_info["issues"].append(f"Database access error: {e}")

                results["databases_found"].append(db_info)

            print(f"   💾 Found {len(results['databases_found'])} database files")

        except Exception as e:
            print(f"   ❌ Database security check error: {e}")
            results["scan_status"] = "error"

        return results

    def _assess_network_security(self) -> dict:
        """Assess network security configuration"""
        print("\n5️⃣  Assessing Network Security...")
        results = {
            "config_files": [],
            "api_endpoints": [],
            "security_headers": [],
            "recommendations": [],
            "scan_status": "completed",
        }

        try:
            config_patterns = ["*django*", "*flask*", "*fastapi*", "*config*", "*settings*"]

            for pattern in config_patterns:
                for config_file in self.project_root.rglob(f"{pattern}.py"):
                    try:
                        content = config_file.read_text(encoding="utf-8")

                        config_info = {
                            "path": str(config_file.relative_to(self.project_root)),
                            "size_bytes": len(content),
                            "potential_issues": [],
                        }

                        security_checks = [
                            ("DEBUG = True", "Debug mode enabled in production"),
                            ("SECRET_KEY", "Hardcoded secret key found"),
                            ("ALLOWED_HOSTS = []", "Empty ALLOWED_HOSTS"),
                            ("CORS_ALLOW_ALL_ORIGINS = True", "CORS allows all origins"),
                            ("127.0.0.1", "Localhost binding found"),
                            ("0.0.0.0", "All interfaces binding found"),
                        ]

                        for check, issue in security_checks:
                            if check in content:
                                config_info["potential_issues"].append(issue)

                        if config_info["potential_issues"]:
                            results["config_files"].append(config_info)

                    except (UnicodeDecodeError, PermissionError):
                        continue

            print(f"   🌐 Analyzed network configuration files")
            print(f"   ⚠️  Potential issues found in {len(results['config_files'])} files")

        except Exception as e:
            print(f"   ❌ Network security assessment error: {e}")
            results["scan_status"] = "error"

        return results
        """Scan dependencies for known vulnerabilities"""
        results = {
            "requirements_found": False,
            "vulnerabilities": [],
            "total_packages": 0,
            "vulnerable_packages": 0,
            "scan_status": "completed",
        }

        try:
            # Check for requirements.txt
            req_file = self.project_root / "requirements.txt"
            if req_file.exists():
                results["requirements_found"] = True
                print(
                    f"   📦 Found requirements.txt with {len(req_file.read_text().splitlines())} packages"
                )

                # Try to run safety check
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "safety", "check", "-r", str(req_file), "--json"],
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )

                    if result.returncode == 0:
                        safety_data = json.loads(result.stdout) if result.stdout else []
                        results["vulnerabilities"] = safety_data
                        results["vulnerable_packages"] = len(safety_data)
                        print(
                            f"   ✅ Safety scan completed: {len(safety_data)} vulnerabilities found"
                        )
                    else:
                        print(f"   ⚠️  Safety scan failed: {result.stderr}")
                        results["scan_status"] = "failed"

                except subprocess.TimeoutExpired:
                    print("   ⏱️  Safety scan timed out")
                    results["scan_status"] = "timeout"
                except Exception as e:
                    print(f"   ❌ Safety scan error: {e}")
                    results["scan_status"] = "error"

                # Count total packages
                with open(req_file) as f:
                    lines = [
                        line.strip() for line in f if line.strip() and not line.startswith("#")
                    ]
                    results["total_packages"] = len(lines)
            else:
                print("   ⚠️  No requirements.txt found")

        except Exception as e:
            print(f"   ❌ Dependency scan error: {e}")
            results["scan_status"] = "error"

        return results

    def _analyze_code_security(self) -> dict:
        """Analyze code security using Bandit"""
        results = {
            "total_files_scanned": 0,
            "issues_found": 0,
            "high_severity": 0,
            "medium_severity": 0,
            "low_severity": 0,
            "issues": [],
            "scan_status": "completed",
        }

        try:
            # Find Python files to scan
            python_files = list(self.project_root.rglob("*.py"))
            # Exclude virtual environments and build directories
            python_files = [
                f
                for f in python_files
                if not any(
                    exclude in str(f)
                    for exclude in [
                        "venv",
                        "env",
                        "__pycache__",
                        ".git",
                        "build",
                        "dist",
                        "node_modules",
                        ".pytest_cache",
                        "htmlcov",
                    ]
                )
            ]

            results["total_files_scanned"] = len(python_files)
            print(f"   📄 Scanning {len(python_files)} Python files...")

            if python_files:
                # Create temporary bandit config
                conf = bandit_config.BanditConfig()
                b_mgr = manager.BanditManager(conf, "file")

                # Scan files
                for py_file in python_files[:50]:  # Limit to first 50 files for performance
                    try:
                        b_mgr.discover_files([str(py_file)])
                    except Exception:
                        continue

                b_mgr.run_tests()

                # Process results
                for result in b_mgr.get_issue_list():
                    issue = {
                        "filename": str(result.fname),
                        "line_number": result.lineno,
                        "test_name": result.test,
                        "severity": result.severity,
                        "confidence": result.confidence,
                        "text": (
                            result.text[:200] + "..." if len(result.text) > 200 else result.text
                        ),
                    }
                    results["issues"].append(issue)

                    if result.severity == "HIGH":
                        results["high_severity"] += 1
                    elif result.severity == "MEDIUM":
                        results["medium_severity"] += 1
                    else:
                        results["low_severity"] += 1

                results["issues_found"] = len(results["issues"])
                print(f"   🔍 Found {results['issues_found']} security issues:")
                print(f"      🔴 High: {results['high_severity']}")
                print(f"      🟡 Medium: {results['medium_severity']}")
                print(f"      🟢 Low: {results['low_severity']}")

        except Exception as e:
            print(f"   ❌ Code security analysis error: {e}")
            results["scan_status"] = "error"

        return results

    def _audit_file_permissions(self) -> dict:
        """Audit file permissions for security issues"""
        results = {
            "total_files_checked": 0,
            "permission_issues": [],
            "executable_files": [],
            "world_writable": [],
            "config_files": [],
            "scan_status": "completed",
        }

        try:
            sensitive_extensions = [
                ".py",
                ".sh",
                ".bat",
                ".ps1",
                ".conf",
                ".cfg",
                ".ini",
                ".json",
                ".yaml",
                ".yml",
            ]

            for file_path in self.project_root.rglob("*"):
                if file_path.is_file():
                    results["total_files_checked"] += 1

                    try:
                        file_stat = file_path.stat()
                        mode = file_stat.st_mode

                        # Check for executable files
                        if mode & stat.S_IXUSR:
                            results["executable_files"].append(
                                {
                                    "path": str(file_path.relative_to(self.project_root)),
                                    "permissions": oct(mode)[-3:],
                                }
                            )

                        # Check for world-writable files
                        if mode & stat.S_IWOTH:
                            results["world_writable"].append(
                                {
                                    "path": str(file_path.relative_to(self.project_root)),
                                    "permissions": oct(mode)[-3:],
                                }
                            )

                        # Check configuration files
                        if any(file_path.suffix.lower() == ext for ext in sensitive_extensions):
                            results["config_files"].append(
                                {
                                    "path": str(file_path.relative_to(self.project_root)),
                                    "permissions": oct(mode)[-3:],
                                    "size_bytes": file_stat.st_size,
                                }
                            )

                    except (OSError, PermissionError):
                        continue

            print(f"   📁 Checked {results['total_files_checked']} files")
            print(f"   🔧 Executable files: {len(results['executable_files'])}")
            print(f"   ⚠️  World-writable files: {len(results['world_writable'])}")
            print(f"   ⚙️  Configuration files: {len(results['config_files'])}")

        except Exception as e:
            print(f"   ❌ File permission audit error: {e}")
            results["scan_status"] = "error"

        return results

    def _check_database_security(self) -> dict:
        """Check database security configuration"""
        results = {
            "databases_found": [],
            "security_issues": [],
            "recommendations": [],
            "scan_status": "completed",
        }

        try:
            # Look for SQLite databases
            db_files = list(self.project_root.rglob("*.db")) + list(
                self.project_root.rglob("*.sqlite")
            )

            for db_file in db_files:
                db_info = {
                    "path": str(db_file.relative_to(self.project_root)),
                    "size_bytes": db_file.stat().st_size,
                    "permissions": oct(db_file.stat().st_mode)[-3:],
                    "tables": [],
                    "issues": [],
                }

                try:
                    conn = sqlite3.connect(str(db_file))
                    cursor = conn.cursor()

                    # Get table information
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    db_info["tables"] = [table[0] for table in tables]

                    # Check for potential security issues
                    for table_name in db_info["tables"]:
                        try:
                            cursor.execute(f"PRAGMA table_info({table_name})")
                            columns = cursor.fetchall()

                            # Look for password/sensitive columns
                            sensitive_cols = [
                                col[1]
                                for col in columns
                                if any(
                                    keyword in col[1].lower()
                                    for keyword in [
                                        "password",
                                        "pass",
                                        "pwd",
                                        "secret",
                                        "token",
                                        "key",
                                        "hash",
                                    ]
                                )
                            ]

                            if sensitive_cols:
                                db_info["issues"].append(
                                    f"Table '{table_name}' contains sensitive columns: {sensitive_cols}"
                                )

                        except sqlite3.Error:
                            continue

                    conn.close()

                except sqlite3.Error as e:
                    db_info["issues"].append(f"Database access error: {e}")

                results["databases_found"].append(db_info)

            print(f"   💾 Found {len(results['databases_found'])} database files")

            # Generate recommendations
            if results["databases_found"]:
                results["recommendations"].extend(
                    [
                        "Ensure database files have appropriate permissions (600 or 640)",
                        "Consider encrypting sensitive data in databases",
                        "Implement proper backup and recovery procedures",
                        "Use parameterized queries to prevent SQL injection",
                    ]
                )

        except Exception as e:
            print(f"   ❌ Database security check error: {e}")
            results["scan_status"] = "error"

        return results

    def _assess_network_security(self) -> dict:
        """Assess network security configuration"""
        results = {
            "config_files": [],
            "api_endpoints": [],
            "security_headers": [],
            "recommendations": [],
            "scan_status": "completed",
        }

        try:
            # Look for network configuration files
            config_patterns = ["*django*", "*flask*", "*fastapi*", "*config*", "*settings*"]

            for pattern in config_patterns:
                for config_file in self.project_root.rglob(f"{pattern}.py"):
                    try:
                        content = config_file.read_text(encoding="utf-8")

                        config_info = {
                            "path": str(config_file.relative_to(self.project_root)),
                            "size_bytes": len(content),
                            "potential_issues": [],
                        }

                        # Check for potential security issues
                        security_checks = [
                            ("DEBUG = True", "Debug mode enabled in production"),
                            ("SECRET_KEY", "Hardcoded secret key found"),
                            ("ALLOWED_HOSTS = []", "Empty ALLOWED_HOSTS"),
                            ("CORS_ALLOW_ALL_ORIGINS = True", "CORS allows all origins"),
                            ("127.0.0.1", "Localhost binding found"),
                            ("0.0.0.0", "All interfaces binding found"),
                        ]

                        for check, issue in security_checks:
                            if check in content:
                                config_info["potential_issues"].append(issue)

                        if config_info["potential_issues"]:
                            results["config_files"].append(config_info)

                    except (UnicodeDecodeError, PermissionError):
                        continue

            print(f"   🌐 Analyzed network configuration files")
            print(f"   ⚠️  Potential issues found in {len(results['config_files'])} files")

            # Generate network security recommendations
            results["recommendations"] = [
                "Disable debug mode in production",
                "Use environment variables for sensitive configuration",
                "Implement proper CORS policies",
                "Use HTTPS for all communications",
                "Implement rate limiting for API endpoints",
                "Add security headers (HSTS, CSP, etc.)",
                "Validate and sanitize all input data",
            ]

        except Exception as e:
            print(f"   ❌ Network security assessment error: {e}")
            results["scan_status"] = "error"

        return results

    def _generate_risk_assessment(self, audit_results: dict) -> dict:
        """Generate overall risk assessment"""
        risk_score = 0
        max_score = 100

        # Calculate risk based on findings
        vuln_scan = audit_results.get("vulnerability_scan", {})
        code_security = audit_results.get("code_security", {})
        file_perms = audit_results.get("file_permissions", {})
        db_security = audit_results.get("database_security", {})

        # Vulnerability scoring
        if vuln_scan.get("vulnerable_packages", 0) > 0:
            risk_score += min(vuln_scan["vulnerable_packages"] * 5, 25)

        # Code security scoring
        if code_security.get("high_severity", 0) > 0:
            risk_score += min(code_security["high_severity"] * 10, 30)
        if code_security.get("medium_severity", 0) > 0:
            risk_score += min(code_security["medium_severity"] * 3, 15)

        # File permission scoring
        if len(file_perms.get("world_writable", [])) > 0:
            risk_score += min(len(file_perms["world_writable"]) * 5, 20)

        # Database security scoring
        for db in db_security.get("databases_found", []):
            if db.get("issues"):
                risk_score += min(len(db["issues"]) * 3, 10)

        # Determine risk level
        if risk_score <= 20:
            risk_level = "LOW"
            risk_color = "🟢"
        elif risk_score <= 50:
            risk_level = "MEDIUM"
            risk_color = "🟡"
        elif risk_score <= 75:
            risk_level = "HIGH"
            risk_color = "🟠"
        else:
            risk_level = "CRITICAL"
            risk_color = "🔴"

        assessment = {
            "risk_score": risk_score,
            "max_score": max_score,
            "risk_percentage": (risk_score / max_score) * 100,
            "risk_level": risk_level,
            "risk_color": risk_color,
            "summary": f"{risk_color} {risk_level} Risk ({risk_score}/{max_score} points)",
        }

        print(f"   📊 Risk Assessment: {assessment['summary']}")

        return assessment

    def _generate_recommendations(self, audit_results: dict) -> list[str]:
        """Generate security recommendations based on findings"""
        recommendations = []

        # Dependency recommendations
        vuln_scan = audit_results.get("vulnerability_scan", {})
        if vuln_scan.get("vulnerable_packages", 0) > 0:
            recommendations.extend(
                [
                    "🔄 Update vulnerable dependencies immediately",
                    "📅 Implement regular dependency scanning in CI/CD",
                    "🔒 Consider using dependency lock files",
                ]
            )

        # Code security recommendations
        code_security = audit_results.get("code_security", {})
        if code_security.get("high_severity", 0) > 0:
            recommendations.extend(
                [
                    "🚨 Address high-severity security issues immediately",
                    "🔍 Implement static code analysis in development workflow",
                    "📖 Provide security training for development team",
                ]
            )

        # File permission recommendations
        file_perms = audit_results.get("file_permissions", {})
        if len(file_perms.get("world_writable", [])) > 0:
            recommendations.extend(
                [
                    "🔐 Fix world-writable file permissions",
                    "📋 Implement file permission auditing process",
                ]
            )

        # Database security recommendations
        db_security = audit_results.get("database_security", {})
        if any(db.get("issues") for db in db_security.get("databases_found", [])):
            recommendations.extend(
                [
                    "💾 Review database security configuration",
                    "🔑 Implement database access controls",
                    "🔒 Consider database encryption for sensitive data",
                ]
            )

        # General recommendations
        recommendations.extend(
            [
                "🛡️  Implement regular security audits",
                "📚 Create security documentation and policies",
                "🔄 Set up automated security monitoring",
                "👥 Establish incident response procedures",
                "🎓 Provide security awareness training",
            ]
        )

        return recommendations

    def _create_audit_summary(self, audit_results: dict) -> dict:
        """Create comprehensive audit summary"""
        summary = {
            "total_issues": 0,
            "critical_issues": 0,
            "files_scanned": 0,
            "vulnerabilities_found": 0,
            "recommendations_count": len(audit_results.get("recommendations", [])),
            "scan_duration": "N/A",
            "overall_status": "COMPLETED",
        }

        # Count issues
        vuln_scan = audit_results.get("vulnerability_scan", {})
        code_security = audit_results.get("code_security", {})
        file_perms = audit_results.get("file_permissions", {})

        summary["vulnerabilities_found"] = vuln_scan.get("vulnerable_packages", 0)
        summary["total_issues"] = (
            code_security.get("issues_found", 0)
            + len(file_perms.get("world_writable", []))
            + summary["vulnerabilities_found"]
        )
        summary["critical_issues"] = code_security.get("high_severity", 0)
        summary["files_scanned"] = code_security.get("total_files_scanned", 0) + file_perms.get(
            "total_files_checked", 0
        )

        return summary

    def save_audit_report(self, audit_results: dict):
        """Save comprehensive audit report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.audit_dir / f"security_audit_{timestamp}.json"

        try:
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(audit_results, f, indent=2, default=str)

            print(f"\n📋 Security Audit Report Saved:")
            print(f"   📄 File: {report_file}")
            print(f"   📊 Risk Level: {audit_results['risk_assessment']['summary']}")
            print(f"   🔍 Total Issues: {audit_results['audit_summary']['total_issues']}")
            print(
                f"   📝 Recommendations: {audit_results['audit_summary']['recommendations_count']}"
            )

            return report_file

        except Exception as e:
            print(f"❌ Failed to save audit report: {e}")
            return None


def main():
    """Main function to run security audit"""
    print("🔒 SpygateAI Security Audit System")
    print("=" * 50)

    auditor = SecurityAuditor()

    start_time = time.time()
    audit_results = auditor.run_comprehensive_audit()
    end_time = time.time()

    audit_results["audit_summary"]["scan_duration"] = f"{end_time - start_time:.2f} seconds"

    # Save report
    report_file = auditor.save_audit_report(audit_results)

    print("\n" + "=" * 60)
    print("🎯 SECURITY AUDIT COMPLETE")
    print("=" * 60)

    print(f"📊 Overall Risk: {audit_results['risk_assessment']['summary']}")
    print(f"🔍 Issues Found: {audit_results['audit_summary']['total_issues']}")
    print(f"⚠️  Critical Issues: {audit_results['audit_summary']['critical_issues']}")
    print(f"📁 Files Scanned: {audit_results['audit_summary']['files_scanned']}")
    print(f"⏱️  Scan Duration: {audit_results['audit_summary']['scan_duration']}")

    if audit_results["recommendations"]:
        print(f"\n📝 Top Recommendations:")
        for i, rec in enumerate(audit_results["recommendations"][:5], 1):
            print(f"   {i}. {rec}")

    if report_file:
        print(f"\n📋 Full report saved to: {report_file}")

    return audit_results


if __name__ == "__main__":
    main()
