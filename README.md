# SpygateAI: ML-Powered Football Gameplay Analysis

SpygateAI is an intelligent engine that automates the discovery and bookmarking of key game situations using HUD analysis in football games.

## Project Structure

```
spygate/
├── src/                    # Source code
│   ├── core/              # Core detection and analysis
│   ├── ui/                # User interface components
│   └── utils/             # Utility functions
├── tests/                 # Test suite
├── data/                  # Data files
│   ├── models/           # Trained models
│   └── test_videos/      # Test video files
└── docs/                 # Documentation
```

## Installation

1. Create a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

2. Install dependencies:

```bash
# For basic installation
pip install -e .

# For development
pip install -e ".[dev,test,docs]"
```

## Development

- Code formatting: `black src tests`
- Type checking: `mypy src`
- Run tests: `pytest`
- Build docs: `cd docs && make html`

## Features

- HUD Detection
- Game State Analysis
- Tendency Tracking
- Real-time Analysis

## License

See LICENSE file for details.

## Overview

Spygate is a Python-based desktop application that uses advanced computer vision (YOLO11) and machine learning to analyze football gameplay, starting with Madden NFL 25. It helps competitive players improve by automatically detecting, categorizing, and organizing game situations for analysis.

## Features

- 🎮 Automated gameplay situation detection
- 🔍 Formation and play recognition
- 📊 Performance analytics and insights
- 🎥 Live stream recording (Twitch/YouTube)
- 🤝 Discord community integration
- 📚 Custom playbook integration
- 🎯 Pro gameplay comparison
- ♿ WCAG 2.1 AA compliant UI

## Requirements

- Python 3.9+
- Windows OS
- NVIDIA GPU recommended for optimal performance
- FFmpeg for video processing

## Installation

1. Clone the repository:

```bash
git clone https://github.com/spygate/spygate.git
cd spygate
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Install pre-commit hooks:

```bash
pre-commit install
```

## Development Setup

1. Install additional development dependencies:

```bash
pip install -r requirements-dev.txt
```

2. Configure environment variables:

```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run tests:

```bash
pytest
```

## Project Structure

```
spygate/
├── src/                    # Source code
│   ├── core/              # Core functionality
│   ├── gui/               # PyQt6 UI components
│   ├── ml/                # Machine learning models
│   ├── video/             # Video processing
│   └── utils/             # Utilities
├── tests/                 # Test files
├── docs/                  # Documentation
├── data/                  # Data files
│   ├── models/           # ML models
│   └── playbooks/        # Playbook data
└── scripts/              # Utility scripts
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure they pass
5. Submit a pull request

## Documentation

- [User Guide](docs/user_guide.md)
- [Developer Guide](docs/developer_guide.md)
- [API Reference](docs/api_reference.md)

## Community

- [Discord Server](https://discord.gg/spygate)
- [Bug Reports](https://github.com/spygate/spygate/issues)
- [Feature Requests](https://github.com/spygate/spygate/issues)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- YOLO11 for object detection
- OpenCV for video processing
- PyQt6 for the GUI framework
- The Madden NFL community for support and feedback

## Development with Docker

### Prerequisites

- [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
- Windows 10 Pro/Enterprise/Education (required for Docker Desktop)
- WSL2 enabled

### Quick Start with Docker

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/spygate.git
   cd spygate
   ```

2. Build and start the containers:

   ```bash
   docker-compose up --build
   ```

3. Access the application:
   - Web interface: http://localhost:8000
   - PostgreSQL: localhost:5432

### Docker Commands

- Start containers in background:

  ```bash
  docker-compose up -d
  ```

- Stop containers:

  ```bash
  docker-compose down
  ```

- View logs:

  ```bash
  docker-compose logs -f
  ```

- Run tests:

  ```bash
  docker-compose exec app pytest
  ```

- Access PostgreSQL:
  ```bash
  docker-compose exec db psql -U postgres -d spygate
  ```

### Development Workflow

1. Make changes to the code
2. Docker will automatically reload the application
3. Run tests inside the container:
   ```bash
   docker-compose exec app pytest
   ```

### Volumes

- `./data`: Persistent data storage
- `./logs`: Application logs
- `./models`: ML model storage
- `postgres_data`: PostgreSQL data (persistent)

### Environment Variables

Default environment variables are set in `docker-compose.yml`. For local development, you can override them by creating a `.env` file.

## Project Size Management

To maintain a clean and efficient project:

1. **Virtual Environments**

   - Use a single virtual environment (`.venv`)
   - Do not commit virtual environments to git
   - Recreate virtual environments using `requirements.txt` or `pyproject.toml`

2. **Large Files**

   - Git LFS is configured for large files (videos, models, datasets)
   - Install Git LFS: `git lfs install`
   - Large files will be automatically tracked based on `.gitattributes`

3. **Test Data**

   - Keep minimal test data in the repository
   - Store large datasets externally (e.g., cloud storage)
   - Use small sample files for testing

4. **Cleanup**

   - Run `cleanup.ps1` periodically to remove:
     - Build artifacts
     - Cache files
     - Temporary files
     - Test artifacts

5. **Best Practices**
   - Remove unused model versions
   - Keep only necessary video recordings
   - Clean up after training sessions
   - Use external storage for archival data

# SpygateAI Web Interface

A modern web interface for SpygateAI, built with Django and React.

## Features

- RESTful API for video analysis and processing
- Real-time progress tracking via WebSockets
- User authentication and authorization
- File upload and management
- Analytics dashboard
- Secure file storage with AWS S3

## Tech Stack

- **Backend**: Django + Django REST Framework + Channels
- **Frontend**: Next.js + React (separate repository)
- **Database**: PostgreSQL
- **Cache & WebSocket**: Redis
- **Storage**: AWS S3
- **Deployment**: Docker + Gunicorn + Nginx

## Development Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/spygate.git
   cd spygate
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   .venv\Scripts\activate     # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

5. Run migrations:

   ```bash
   cd spygate_site
   python manage.py migrate
   ```

6. Create a superuser:

   ```bash
   python manage.py createsuperuser
   ```

7. Run the development server:
   ```bash
   python manage.py runserver
   ```

## Docker Development Setup

1. Build and start the containers:

   ```bash
   docker-compose up --build
   ```

2. Run migrations:

   ```bash
   docker-compose exec web python spygate_site/manage.py migrate
   ```

3. Create a superuser:
   ```bash
   docker-compose exec web python spygate_site/manage.py createsuperuser
   ```

## Production Deployment

1. Set up environment variables:

   ```bash
   cp .env.example .env.prod
   # Edit .env.prod with production settings
   ```

2. Build the production image:

   ```bash
   docker build -t spygate-web .
   ```

3. Run the container:
   ```bash
   docker run -d \
     --env-file .env.prod \
     -p 8000:8000 \
     spygate-web
   ```

## API Documentation

API documentation is available at `/api/docs/` when running the development server.

## Testing

Run the test suite:

```bash
pytest
```

Run with coverage:

```bash
coverage run -m pytest
coverage report
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
