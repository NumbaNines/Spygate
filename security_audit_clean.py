#!/usr/bin/env python3
"""
SpygateAI Security Audit System - Step 11
Enterprise-grade security scanning and vulnerability assessment
"""

import json
import os
import sqlite3
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "safety"])
    import requests


class SecurityAuditor:
    """Comprehensive security audit system for SpygateAI"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.audit_dir = self.project_root / "security"
        self.audit_dir.mkdir(exist_ok=True)

        print("🔒 SpygateAI Security Auditor Initialized")
        print(f"📁 Project Root: {self.project_root.absolute()}")

    def run_comprehensive_audit(self) -> dict:
        """Run complete security audit"""
        print("\n🔍 Starting Comprehensive Security Audit...")
        print("=" * 60)

        audit_results = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root.absolute()),
            "vulnerability_scan": self._scan_dependencies(),
            "file_permissions": self._audit_file_permissions(),
            "database_security": self._check_database_security(),
            "network_security": self._assess_network_security(),
        }

        # Generate analysis
        audit_results["risk_assessment"] = self._generate_risk_assessment(audit_results)
        audit_results["recommendations"] = self._generate_recommendations(audit_results)
        audit_results["audit_summary"] = self._create_audit_summary(audit_results)

        return audit_results

    def _scan_dependencies(self) -> dict:
        """Scan dependencies for vulnerabilities"""
        print("\n1️⃣  Scanning Dependencies...")
        results = {
            "requirements_found": False,
            "total_packages": 0,
            "scan_status": "completed",
            "issues": [],
        }

        try:
            req_file = self.project_root / "requirements.txt"
            if req_file.exists():
                results["requirements_found"] = True
                lines = req_file.read_text().splitlines()
                results["total_packages"] = len(
                    [l for l in lines if l.strip() and not l.startswith("#")]
                )
                print(f"   📦 Found {results['total_packages']} packages")

                # Basic dependency analysis
                for line in lines:
                    if line.strip() and not line.startswith("#"):
                        if "==" not in line and ">=" not in line:
                            results["issues"].append(f"Unpinned dependency: {line.strip()}")
            else:
                print("   ⚠️  No requirements.txt found")
                results["issues"].append("No requirements.txt file found")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results["scan_status"] = "error"

        return results

    def _audit_file_permissions(self) -> dict:
        """Audit file permissions"""
        print("\n2️⃣  Auditing File Permissions...")
        results = {
            "total_files_checked": 0,
            "executable_files": [],
            "config_files": [],
            "sensitive_files": [],
            "scan_status": "completed",
        }

        try:
            sensitive_extensions = [".py", ".json", ".yaml", ".yml", ".conf", ".cfg", ".ini"]

            for file_path in self.project_root.rglob("*"):
                if file_path.is_file() and not any(
                    exclude in str(file_path)
                    for exclude in ["__pycache__", ".git", "node_modules", ".pytest_cache"]
                ):

                    results["total_files_checked"] += 1

                    try:
                        file_stat = file_path.stat()
                        mode = file_stat.st_mode
                        rel_path = str(file_path.relative_to(self.project_root))

                        # Check executable files
                        if mode & stat.S_IXUSR:
                            results["executable_files"].append(
                                {"path": rel_path, "permissions": oct(mode)[-3:]}
                            )

                        # Check sensitive config files
                        if any(file_path.suffix.lower() == ext for ext in sensitive_extensions):
                            results["config_files"].append(
                                {
                                    "path": rel_path,
                                    "permissions": oct(mode)[-3:],
                                    "size_bytes": file_stat.st_size,
                                }
                            )

                        # Check for sensitive content
                        if file_path.suffix.lower() in [".py", ".json", ".yaml", ".yml"]:
                            try:
                                content = file_path.read_text(encoding="utf-8", errors="ignore")
                                if any(
                                    keyword in content.lower()
                                    for keyword in ["password", "secret", "api_key", "token"]
                                ):
                                    results["sensitive_files"].append(
                                        {"path": rel_path, "reason": "Contains sensitive keywords"}
                                    )
                            except:
                                continue

                    except (OSError, PermissionError):
                        continue

            print(f"   📁 Checked {results['total_files_checked']} files")
            print(f"   🔧 Executable: {len(results['executable_files'])}")
            print(f"   ⚙️  Config files: {len(results['config_files'])}")
            print(f"   🔑 Sensitive: {len(results['sensitive_files'])}")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results["scan_status"] = "error"

        return results

    def _check_database_security(self) -> dict:
        """Check database security"""
        print("\n3️⃣  Checking Database Security...")
        results = {"databases_found": [], "security_issues": [], "scan_status": "completed"}

        try:
            db_files = list(self.project_root.rglob("*.db")) + list(
                self.project_root.rglob("*.sqlite")
            )

            for db_file in db_files:
                db_info = {
                    "path": str(db_file.relative_to(self.project_root)),
                    "size_bytes": db_file.stat().st_size,
                    "tables": [],
                    "issues": [],
                }

                try:
                    conn = sqlite3.connect(str(db_file))
                    cursor = conn.cursor()

                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    db_info["tables"] = [table[0] for table in tables]

                    # Check for sensitive data
                    for table_name in db_info["tables"][:10]:  # Limit for performance
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
                                    ]
                                )
                            ]

                            if sensitive_cols:
                                db_info["issues"].append(
                                    f"Sensitive columns in {table_name}: {sensitive_cols}"
                                )

                        except sqlite3.Error:
                            continue

                    conn.close()

                except sqlite3.Error as e:
                    db_info["issues"].append(f"Access error: {str(e)}")

                results["databases_found"].append(db_info)

            print(f"   💾 Found {len(results['databases_found'])} databases")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results["scan_status"] = "error"

        return results

    def _assess_network_security(self) -> dict:
        """Assess network security"""
        print("\n4️⃣  Assessing Network Security...")
        results = {"config_files": [], "potential_issues": [], "scan_status": "completed"}

        try:
            # Check Django/Flask settings
            for config_file in self.project_root.rglob("*settings*.py"):
                try:
                    content = config_file.read_text(encoding="utf-8")
                    rel_path = str(config_file.relative_to(self.project_root))

                    issues = []
                    if "DEBUG = True" in content:
                        issues.append("Debug mode enabled")
                    if "SECRET_KEY" in content and not "os.environ" in content:
                        issues.append("Hardcoded secret key")
                    if "ALLOWED_HOSTS = []" in content:
                        issues.append("Empty ALLOWED_HOSTS")

                    if issues:
                        results["config_files"].append({"path": rel_path, "issues": issues})

                except (UnicodeDecodeError, PermissionError):
                    continue

            print(f"   🌐 Checked network configuration")
            print(f"   ⚠️  Issues in {len(results['config_files'])} files")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            results["scan_status"] = "error"

        return results

    def _generate_risk_assessment(self, audit_results: dict) -> dict:
        """Generate risk assessment"""
        print("\n6️⃣  Generating Risk Assessment...")

        risk_score = 0

        # Dependency issues
        dep_issues = len(audit_results.get("vulnerability_scan", {}).get("issues", []))
        risk_score += min(dep_issues * 5, 20)

        # File permission issues
        sensitive_files = len(audit_results.get("file_permissions", {}).get("sensitive_files", []))
        risk_score += min(sensitive_files * 3, 15)

        # Database issues
        for db in audit_results.get("database_security", {}).get("databases_found", []):
            risk_score += len(db.get("issues", [])) * 2

        # Network security issues
        for config in audit_results.get("network_security", {}).get("config_files", []):
            risk_score += len(config.get("issues", [])) * 3

        # Determine risk level
        if risk_score <= 10:
            risk_level, color = "LOW", "🟢"
        elif risk_score <= 25:
            risk_level, color = "MEDIUM", "🟡"
        elif risk_score <= 40:
            risk_level, color = "HIGH", "🟠"
        else:
            risk_level, color = "CRITICAL", "🔴"

        assessment = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_color": color,
            "summary": f"{color} {risk_level} Risk ({risk_score} points)",
        }

        print(f"   📊 {assessment['summary']}")
        return assessment

    def _generate_recommendations(self, audit_results: dict) -> list[str]:
        """Generate security recommendations"""
        print("\n7️⃣  Generating Recommendations...")

        recommendations = []

        # Dependency recommendations
        dep_scan = audit_results.get("vulnerability_scan", {})
        if dep_scan.get("issues"):
            recommendations.extend(
                [
                    "🔄 Pin all dependency versions in requirements.txt",
                    "📅 Implement regular dependency scanning",
                    "🔒 Use virtual environments for isolation",
                ]
            )

        # File security recommendations
        file_perms = audit_results.get("file_permissions", {})
        if file_perms.get("sensitive_files"):
            recommendations.extend(
                [
                    "🔐 Remove hardcoded secrets from source code",
                    "🌍 Use environment variables for sensitive data",
                    "📋 Implement .gitignore for sensitive files",
                ]
            )

        # Database recommendations
        db_security = audit_results.get("database_security", {})
        if any(db.get("issues") for db in db_security.get("databases_found", [])):
            recommendations.extend(
                [
                    "💾 Review database security configuration",
                    "🔑 Implement proper access controls",
                    "🔒 Consider encryption for sensitive data",
                ]
            )

        # Network security recommendations
        net_security = audit_results.get("network_security", {})
        if net_security.get("config_files"):
            recommendations.extend(
                [
                    "🚫 Disable debug mode in production",
                    "🔑 Use environment variables for secrets",
                    "🌐 Configure proper ALLOWED_HOSTS",
                ]
            )

        # General recommendations
        recommendations.extend(
            [
                "🛡️  Implement regular security audits",
                "📚 Create security documentation",
                "🔄 Set up automated security monitoring",
                "🎓 Provide security training",
            ]
        )

        print(f"   📝 Generated {len(recommendations)} recommendations")
        return recommendations

    def _create_audit_summary(self, audit_results: dict) -> dict:
        """Create audit summary"""
        summary = {
            "total_issues": 0,
            "files_scanned": 0,
            "recommendations_count": len(audit_results.get("recommendations", [])),
            "overall_status": "COMPLETED",
        }

        # Count issues
        dep_issues = len(audit_results.get("vulnerability_scan", {}).get("issues", []))
        file_issues = len(audit_results.get("file_permissions", {}).get("sensitive_files", []))

        db_issues = sum(
            len(db.get("issues", []))
            for db in audit_results.get("database_security", {}).get("databases_found", [])
        )

        net_issues = sum(
            len(config.get("issues", []))
            for config in audit_results.get("network_security", {}).get("config_files", [])
        )

        summary["total_issues"] = dep_issues + file_issues + db_issues + net_issues
        summary["files_scanned"] = audit_results.get("file_permissions", {}).get(
            "total_files_checked", 0
        )

        return summary

    def save_audit_report(self, audit_results: dict):
        """Save audit report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.audit_dir / f"security_audit_{timestamp}.json"

        try:
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(audit_results, f, indent=2, default=str)

            print(f"\n📋 Security Audit Report Saved:")
            print(f"   📄 File: {report_file}")
            print(f"   📊 Risk: {audit_results['risk_assessment']['summary']}")
            print(f"   🔍 Issues: {audit_results['audit_summary']['total_issues']}")

            return report_file

        except Exception as e:
            print(f"❌ Failed to save report: {e}")
            return None


def main():
    """Main function"""
    print("🔒 SpygateAI Security Audit System - Step 11")
    print("=" * 50)

    auditor = SecurityAuditor()

    start_time = time.time()
    audit_results = auditor.run_comprehensive_audit()
    audit_results["audit_summary"]["scan_duration"] = f"{time.time() - start_time:.2f}s"

    # Save report
    report_file = auditor.save_audit_report(audit_results)

    print("\n" + "=" * 60)
    print("🎯 SECURITY AUDIT COMPLETE")
    print("=" * 60)

    summary = audit_results["audit_summary"]
    print(f"📊 Risk: {audit_results['risk_assessment']['summary']}")
    print(f"🔍 Issues: {summary['total_issues']}")
    print(f"📁 Files: {summary['files_scanned']}")
    print(f"⏱️  Duration: {summary['scan_duration']}")

    if audit_results["recommendations"]:
        print(f"\n📝 Top Recommendations:")
        for i, rec in enumerate(audit_results["recommendations"][:5], 1):
            print(f"   {i}. {rec}")

    return audit_results


if __name__ == "__main__":
    main()
