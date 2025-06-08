from setuptools import find_packages, setup

setup(
    name="spygate",
    version="0.1.0",
    description="ML-powered football gameplay analysis tool",
    author="SpygateAI Team",
    packages=find_packages(),
    install_requires=[
        # UI Dependencies
        "PyQt6>=6.9.0",
        # Core ML/CV Dependencies
        "opencv-python>=4.9.0",
        "numpy>=1.26.0",
        "torch>=2.2.0",
        "ultralytics>=8.1.0",  # YOLO
        # Video Processing
        "ffmpeg-python>=0.2.0",
        "streamlink>=6.5.0",
        # Database
        "SQLAlchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",  # PostgreSQL
        "alembic>=1.13.0",  # DB migrations
        # Error Tracking & Monitoring
        "sentry-sdk>=1.40.0",
        "python-json-logger>=2.0.0",
        # Testing
        "pytest>=8.4.0",
        "pytest-qt>=4.4.0",
        "pytest-cov>=6.1.0",
        "pytest-asyncio>=0.23.0",
        "pytest-benchmark>=4.0.0",
        # Development Tools
        "black>=24.1.0",
        "flake8>=7.0.0",
        "mypy>=1.8.0",
        "isort>=5.13.0",
    ],
    extras_require={
        "dev": [
            "pre-commit>=3.6.0",
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=2.0.0",
        ],
        "gpu": [
            "torch-cuda>=2.2.0",
            "onnxruntime-gpu>=1.17.0",
        ],
    },
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "spygate=spygate.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Programming Language :: Python :: 3.9",
        "Topic :: Games/Entertainment :: Sports",
    ],
)
