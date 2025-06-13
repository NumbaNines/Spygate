# SpygateAI Security Configuration Template
# Implement these settings in your application

class SecurityConfig:
    # Authentication Settings
    REQUIRE_MFA = True
    SESSION_TIMEOUT = 1800  # 30 minutes
    MAX_LOGIN_ATTEMPTS = 5
    
    # Password Requirements
    PASSWORD_MIN_LENGTH = 12
    PASSWORD_REQUIRE_SPECIAL = True
    
    # Session Security
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    
    # CSRF Protection
    CSRF_ENABLED = True
    CSRF_TIME_LIMIT = 3600
    
    # Security Headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains'
    }
