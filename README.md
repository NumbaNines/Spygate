# Spygate

AI-Powered Football Gameplay Analysis Tool

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
