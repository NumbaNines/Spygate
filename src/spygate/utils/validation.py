"""
Comprehensive Input Validation and Security System for SpygateAI.

This module provides robust input validation, sanitization, and security
measures to prevent common vulnerabilities and ensure data integrity.
"""

import re
import os
import mimetypes
from pathlib import Path
from typing import Any, List, Dict, Optional, Union, Callable
from urllib.parse import urlparse
import json

from .error_handling import ValidationError, get_error_handler
from .logging_config import get_logger


class SecurityValidator:
    """Security-focused input validation."""
    
    def __init__(self):
        self.logger = get_logger()
        self.error_handler = get_error_handler()
        
        # Security patterns
        self.sql_injection_patterns = [
            r"(\bUNION\b.*\bSELECT\b)",
            r"(\bSELECT\b.*\bFROM\b)",
            r"(\bINSERT\b.*\bINTO\b)",
            r"(\bUPDATE\b.*\bSET\b)",
            r"(\bDELETE\b.*\bFROM\b)",
            r"(\bDROP\b.*\bTABLE\b)",
            r"(\bALTER\b.*\bTABLE\b)",
            r"(--|\#|\/\*|\*\/)",
            r"(\bOR\b.*=.*)",
            r"(\bAND\b.*=.*)",
            r"(;.*--)",
            r"(xp_.*\()",
            r"(sp_.*\()"
        ]
        
        self.xss_patterns = [
            r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>",
            r"javascript:",
            r"vbscript:",
            r"onload=",
            r"onerror=",
            r"onclick=",
            r"onmouseover=",
            r"<iframe\b",
            r"<object\b",
            r"<embed\b",
            r"<applet\b"
        ]
        
        self.path_traversal_patterns = [
            r"\.\.\/",
            r"\.\.\\",
            r"\/\.\.",
            r"\\\.\.",
            r"%2e%2e%2f",
            r"%2e%2e%5c",
            r"%2f%2e%2e",
            r"%5c%2e%2e"
        ]
        
        # Allowed file extensions by category
        self.allowed_extensions = {
            "video": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"],
            "image": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"],
            "audio": [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a"],
            "document": [".pdf", ".txt", ".md", ".doc", ".docx"],
            "data": [".json", ".csv", ".xml", ".yaml", ".yml"],
            "archive": [".zip", ".rar", ".7z", ".tar", ".gz"]
        }
        
        # Maximum file sizes (in bytes)
        self.max_file_sizes = {
            "video": 5 * 1024 * 1024 * 1024,  # 5GB
            "image": 50 * 1024 * 1024,        # 50MB
            "audio": 100 * 1024 * 1024,       # 100MB
            "document": 10 * 1024 * 1024,     # 10MB
            "data": 5 * 1024 * 1024,          # 5MB
            "archive": 100 * 1024 * 1024      # 100MB
        }
    
    def validate_string_input(
        self, 
        value: str, 
        field_name: str = "input",
        min_length: int = 0,
        max_length: int = 1000,
        allow_empty: bool = True,
        pattern: Optional[str] = None,
        check_sql_injection: bool = True,
        check_xss: bool = True
    ) -> str:
        """Validate and sanitize string input."""
        if value is None:
            if not allow_empty:
                raise ValidationError(f"{field_name} cannot be None")
            return ""
        
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string, got {type(value)}")
        
        # Length validation
        if len(value) < min_length:
            raise ValidationError(f"{field_name} must be at least {min_length} characters long")
        
        if len(value) > max_length:
            raise ValidationError(f"{field_name} cannot exceed {max_length} characters")
        
        # Empty validation
        if not allow_empty and not value.strip():
            raise ValidationError(f"{field_name} cannot be empty")
        
        # Pattern validation
        if pattern and not re.match(pattern, value):
            raise ValidationError(f"{field_name} does not match required pattern")
        
        # Security checks
        if check_sql_injection and self._contains_sql_injection(value):
            raise ValidationError(f"{field_name} contains potentially malicious SQL patterns")
        
        if check_xss and self._contains_xss(value):
            raise ValidationError(f"{field_name} contains potentially malicious script patterns")
        
        return value.strip()
    
    def validate_email(self, email: str) -> str:
        """Validate email address."""
        email = self.validate_string_input(
            email, 
            "email",
            min_length=3,
            max_length=254,
            allow_empty=False,
            pattern=r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        )
        
        # Additional email validations
        if email.count('@') != 1:
            raise ValidationError("Email must contain exactly one @ symbol")
        
        local, domain = email.split('@')
        
        if len(local) > 64:
            raise ValidationError("Email local part cannot exceed 64 characters")
        
        if len(domain) > 253:
            raise ValidationError("Email domain cannot exceed 253 characters")
        
        return email.lower()
    
    def validate_username(self, username: str) -> str:
        """Validate username."""
        username = self.validate_string_input(
            username,
            "username",
            min_length=3,
            max_length=30,
            allow_empty=False,
            pattern=r'^[a-zA-Z0-9_-]+$'
        )
        
        # Additional username rules
        if username.startswith(('_', '-')) or username.endswith(('_', '-')):
            raise ValidationError("Username cannot start or end with underscore or hyphen")
        
        # Reserved usernames
        reserved = ['admin', 'root', 'system', 'user', 'test', 'guest', 'anonymous']
        if username.lower() in reserved:
            raise ValidationError(f"Username '{username}' is reserved")
        
        return username
    
    def validate_password(self, password: str) -> str:
        """Validate password strength."""
        if not password:
            raise ValidationError("Password is required")
        
        if len(password) < 8:
            raise ValidationError("Password must be at least 8 characters long")
        
        if len(password) > 128:
            raise ValidationError("Password cannot exceed 128 characters")
        
        # Strength requirements
        has_upper = any(c.isupper() for c in password)
        has_lower = any(c.islower() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
        
        strength_checks = [has_upper, has_lower, has_digit, has_special]
        strength_score = sum(strength_checks)
        
        if strength_score < 3:
            raise ValidationError(
                "Password must contain at least 3 of: uppercase letter, lowercase letter, digit, special character"
            )
        
        # Common password check
        common_passwords = [
            "password", "123456", "123456789", "qwerty", "abc123", "password123",
            "admin", "letmein", "welcome", "monkey", "1234567890"
        ]
        
        if password.lower() in common_passwords:
            raise ValidationError("Password is too common, please choose a stronger password")
        
        return password
    
    def validate_file_path(
        self, 
        file_path: Union[str, Path],
        must_exist: bool = True,
        file_category: Optional[str] = None,
        check_size: bool = True,
        check_traversal: bool = True
    ) -> Path:
        """Validate file path and security."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        if not isinstance(file_path, Path):
            raise ValidationError(f"File path must be string or Path object, got {type(file_path)}")
        
        # Convert to absolute path
        file_path = file_path.resolve()
        
        # Path traversal check
        if check_traversal and self._contains_path_traversal(str(file_path)):
            raise ValidationError("File path contains path traversal sequences")
        
        # Existence check
        if must_exist and not file_path.exists():
            raise ValidationError(f"File does not exist: {file_path}")
        
        if must_exist and not file_path.is_file():
            raise ValidationError(f"Path is not a file: {file_path}")
        
        # Extension validation
        if file_category and file_category in self.allowed_extensions:
            allowed_exts = self.allowed_extensions[file_category]
            if file_path.suffix.lower() not in allowed_exts:
                raise ValidationError(
                    f"File extension '{file_path.suffix}' not allowed for {file_category}. "
                    f"Allowed: {allowed_exts}"
                )
        
        # Size validation
        if check_size and must_exist and file_category:
            file_size = file_path.stat().st_size
            max_size = self.max_file_sizes.get(file_category, 100 * 1024 * 1024)  # Default 100MB
            
            if file_size > max_size:
                raise ValidationError(
                    f"File size ({file_size / (1024*1024):.1f}MB) exceeds maximum "
                    f"allowed for {file_category} ({max_size / (1024*1024):.1f}MB)"
                )
        
        # MIME type validation
        if must_exist and file_category:
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                expected_types = {
                    "video": ["video/"],
                    "image": ["image/"],
                    "audio": ["audio/"],
                    "document": ["text/", "application/pdf"],
                    "data": ["text/", "application/json", "application/xml"],
                    "archive": ["application/zip", "application/x-rar"]
                }
                
                if file_category in expected_types:
                    valid_type = any(
                        mime_type.startswith(prefix) 
                        for prefix in expected_types[file_category]
                    )
                    
                    if not valid_type:
                        self.logger.warning(f"MIME type mismatch: {mime_type} for {file_category} file")
        
        return file_path
    
    def validate_url(self, url: str, allowed_schemes: List[str] = None) -> str:
        """Validate URL."""
        if not url:
            raise ValidationError("URL cannot be empty")
        
        url = self.validate_string_input(url, "URL", max_length=2048)
        
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {e}")
        
        if not parsed.scheme:
            raise ValidationError("URL must include a scheme (http, https, etc.)")
        
        if not parsed.netloc:
            raise ValidationError("URL must include a domain")
        
        # Scheme validation
        if allowed_schemes is None:
            allowed_schemes = ["http", "https"]
        
        if parsed.scheme.lower() not in allowed_schemes:
            raise ValidationError(f"URL scheme '{parsed.scheme}' not allowed. Allowed: {allowed_schemes}")
        
        # Block local/private URLs for security
        if parsed.netloc.lower() in ["localhost", "127.0.0.1", "0.0.0.0"]:
            raise ValidationError("Local URLs not allowed")
        
        if parsed.netloc.startswith("192.168.") or parsed.netloc.startswith("10."):
            raise ValidationError("Private network URLs not allowed")
        
        return url
    
    def validate_json_input(self, json_str: str, max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """Validate and parse JSON input."""
        if not json_str:
            raise ValidationError("JSON input cannot be empty")
        
        if len(json_str) > max_size:
            raise ValidationError(f"JSON input too large (max {max_size} bytes)")
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON format: {e}")
        
        if not isinstance(data, dict):
            raise ValidationError("JSON input must be an object")
        
        return data
    
    def validate_numeric_input(
        self,
        value: Union[int, float, str],
        field_name: str = "numeric input",
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None,
        allow_float: bool = True
    ) -> Union[int, float]:
        """Validate numeric input."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValidationError(f"{field_name} cannot be empty")
            
            try:
                if allow_float and '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                raise ValidationError(f"{field_name} must be a valid number")
        
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name} must be a number, got {type(value)}")
        
        if not allow_float and isinstance(value, float):
            raise ValidationError(f"{field_name} must be an integer")
        
        if min_value is not None and value < min_value:
            raise ValidationError(f"{field_name} must be at least {min_value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"{field_name} cannot exceed {max_value}")
        
        return value
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal and illegal characters."""
        if not filename:
            raise ValidationError("Filename cannot be empty")
        
        # Remove path components
        filename = os.path.basename(filename)
        
        # Remove illegal characters
        illegal_chars = '<>:"/\\|?*'
        for char in illegal_chars:
            filename = filename.replace(char, '_')
        
        # Remove control characters
        filename = ''.join(char for char in filename if ord(char) >= 32)
        
        # Remove leading/trailing spaces and dots
        filename = filename.strip(' .')
        
        # Check for reserved names (Windows)
        reserved_names = ['CON', 'PRN', 'AUX', 'NUL'] + [f'COM{i}' for i in range(1, 10)] + [f'LPT{i}' for i in range(1, 10)]
        name_without_ext = os.path.splitext(filename)[0].upper()
        
        if name_without_ext in reserved_names:
            filename = f"file_{filename}"
        
        # Ensure minimum length
        if len(filename) < 1:
            filename = "unnamed_file"
        
        # Ensure maximum length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
        
        return filename
    
    def _contains_sql_injection(self, text: str) -> bool:
        """Check for SQL injection patterns."""
        text_upper = text.upper()
        for pattern in self.sql_injection_patterns:
            if re.search(pattern, text_upper, re.IGNORECASE):
                return True
        return False
    
    def _contains_xss(self, text: str) -> bool:
        """Check for XSS patterns."""
        for pattern in self.xss_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _contains_path_traversal(self, text: str) -> bool:
        """Check for path traversal patterns."""
        for pattern in self.path_traversal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


class DataValidator:
    """Application-specific data validation."""
    
    def __init__(self):
        self.security_validator = SecurityValidator()
        self.logger = get_logger()
    
    def validate_video_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Validate video processing settings."""
        validated = {}
        
        # Frame rate
        if "fps" in settings:
            validated["fps"] = self.security_validator.validate_numeric_input(
                settings["fps"], "FPS", min_value=1, max_value=120, allow_float=True
            )
        
        # Resolution
        if "width" in settings:
            validated["width"] = self.security_validator.validate_numeric_input(
                settings["width"], "width", min_value=128, max_value=7680, allow_float=False
            )
        
        if "height" in settings:
            validated["height"] = self.security_validator.validate_numeric_input(
                settings["height"], "height", min_value=128, max_value=4320, allow_float=False
            )
        
        # Quality
        if "quality" in settings:
            validated["quality"] = self.security_validator.validate_numeric_input(
                settings["quality"], "quality", min_value=0.1, max_value=1.0, allow_float=True
            )
        
        # Codec
        if "codec" in settings:
            allowed_codecs = ["h264", "h265", "vp9", "av1"]
            codec = self.security_validator.validate_string_input(
                settings["codec"], "codec", max_length=10
            ).lower()
            
            if codec not in allowed_codecs:
                raise ValidationError(f"Codec '{codec}' not supported. Allowed: {allowed_codecs}")
            
            validated["codec"] = codec
        
        return validated
    
    def validate_analysis_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Validate analysis settings."""
        validated = {}
        
        # Confidence threshold
        if "confidence_threshold" in settings:
            validated["confidence_threshold"] = self.security_validator.validate_numeric_input(
                settings["confidence_threshold"], "confidence threshold", 
                min_value=0.0, max_value=1.0, allow_float=True
            )
        
        # Batch size
        if "batch_size" in settings:
            validated["batch_size"] = self.security_validator.validate_numeric_input(
                settings["batch_size"], "batch size", 
                min_value=1, max_value=64, allow_float=False
            )
        
        # Model name
        if "model_name" in settings:
            allowed_models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
            model = self.security_validator.validate_string_input(
                settings["model_name"], "model name", max_length=20
            ).lower()
            
            if model not in allowed_models:
                raise ValidationError(f"Model '{model}' not supported. Allowed: {allowed_models}")
            
            validated["model_name"] = model
        
        # Device
        if "device" in settings:
            allowed_devices = ["cpu", "cuda", "auto"]
            device = self.security_validator.validate_string_input(
                settings["device"], "device", max_length=10
            ).lower()
            
            if device not in allowed_devices:
                raise ValidationError(f"Device '{device}' not supported. Allowed: {allowed_devices}")
            
            validated["device"] = device
        
        return validated
    
    def validate_user_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Validate user profile data."""
        validated = {}
        
        if "username" in profile:
            validated["username"] = self.security_validator.validate_username(profile["username"])
        
        if "email" in profile:
            validated["email"] = self.security_validator.validate_email(profile["email"])
        
        if "display_name" in profile:
            validated["display_name"] = self.security_validator.validate_string_input(
                profile["display_name"], "display name", min_length=1, max_length=50
            )
        
        if "bio" in profile:
            validated["bio"] = self.security_validator.validate_string_input(
                profile["bio"], "bio", max_length=500, allow_empty=True
            )
        
        if "profile_picture" in profile:
            # Validate emoji or file path
            pic = profile["profile_picture"]
            if len(pic) <= 4:  # Likely an emoji
                validated["profile_picture"] = pic
                validated["profile_picture_type"] = "emoji"
            else:
                # File path validation would happen during upload
                validated["profile_picture"] = self.security_validator.validate_string_input(
                    pic, "profile picture", max_length=255
                )
                validated["profile_picture_type"] = "custom"
        
        return validated


# Global validator instances
_security_validator: Optional[SecurityValidator] = None
_data_validator: Optional[DataValidator] = None


def get_security_validator() -> SecurityValidator:
    """Get or create the global security validator instance."""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator()
    return _security_validator


def get_data_validator() -> DataValidator:
    """Get or create the global data validator instance."""
    global _data_validator
    if _data_validator is None:
        _data_validator = DataValidator()
    return _data_validator


# Convenience validation functions
def validate_file_upload(file_path: Union[str, Path], file_category: str) -> Path:
    """Validate file upload."""
    return get_security_validator().validate_file_path(
        file_path, must_exist=True, file_category=file_category, check_size=True
    )


def validate_user_input(data: Dict[str, Any], validation_rules: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate user input against rules."""
    validator = get_security_validator()
    validated = {}
    
    for field, rules in validation_rules.items():
        if field in data:
            value = data[field]
            
            if rules.get("type") == "string":
                validated[field] = validator.validate_string_input(
                    value, field, 
                    min_length=rules.get("min_length", 0),
                    max_length=rules.get("max_length", 1000),
                    allow_empty=rules.get("allow_empty", True),
                    pattern=rules.get("pattern")
                )
            elif rules.get("type") == "email":
                validated[field] = validator.validate_email(value)
            elif rules.get("type") == "number":
                validated[field] = validator.validate_numeric_input(
                    value, field,
                    min_value=rules.get("min_value"),
                    max_value=rules.get("max_value"),
                    allow_float=rules.get("allow_float", True)
                )
            elif rules.get("type") == "file":
                validated[field] = validator.validate_file_path(
                    value, 
                    must_exist=rules.get("must_exist", True),
                    file_category=rules.get("category")
                )
            else:
                validated[field] = value
        elif rules.get("required", False):
            raise ValidationError(f"Required field '{field}' is missing")
    
    return validated


if __name__ == "__main__":
    # Test the validation system
    validator = get_security_validator()
    data_validator = get_data_validator()
    logger = get_logger()
    
    logger.info("Testing validation system")
    
    # Test string validation
    try:
        result = validator.validate_string_input("test input", "test", min_length=5, max_length=50)
        logger.info(f"String validation passed: {result}")
    except ValidationError as e:
        logger.error(f"String validation failed: {e}")
    
    # Test email validation
    try:
        email = validator.validate_email("test@example.com")
        logger.info(f"Email validation passed: {email}")
    except ValidationError as e:
        logger.error(f"Email validation failed: {e}")
    
    # Test password validation
    try:
        password = validator.validate_password("StrongPass123!")
        logger.info("Password validation passed")
    except ValidationError as e:
        logger.error(f"Password validation failed: {e}")
    
    # Test user profile validation
    profile = {
        "username": "testuser",
        "email": "test@example.com",
        "display_name": "Test User",
        "bio": "This is a test bio"
    }
    
    try:
        validated_profile = data_validator.validate_user_profile(profile)
        logger.info(f"Profile validation passed: {validated_profile}")
    except ValidationError as e:
        logger.error(f"Profile validation failed: {e}")
    
    logger.info("Validation tests completed")