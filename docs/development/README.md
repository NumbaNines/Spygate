# Development Guide

This guide covers everything you need to know to start developing Spygate.

## Development Environment Setup

1. **Prerequisites**

   - Python 3.10+
   - Git
   - FFmpeg
   - Your favorite IDE (VS Code recommended)
   - Docker (optional, for containerized development)

2. **Clone and Setup**

   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/spygate.git
   cd spygate

   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   # Windows:
   .venv\Scripts\activate
   # Unix/MacOS:
   source .venv/bin/activate

   # Install development dependencies
   pip install -r requirements-dev.txt
   ```

3. **IDE Configuration**

   VS Code Settings (`.vscode/settings.json`):

   ```json
   {
     "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
     "python.linting.enabled": true,
     "python.linting.pylintEnabled": true,
     "python.formatting.provider": "black",
     "editor.formatOnSave": true,
     "editor.codeActionsOnSave": {
       "source.organizeImports": true
     }
   }
   ```

## Code Style

We follow these style guides:

- PEP 8 for Python code style
- Black for code formatting
- isort for import sorting
- Pylint for linting

Pre-commit hooks are configured to enforce these standards:

```bash
# Install pre-commit hooks
pre-commit install
```

## Testing

1. **Running Tests**

   ```bash
   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=spygate

   # Run specific test file
   pytest tests/test_video_service.py
   ```

2. **Writing Tests**
   - Use pytest fixtures for setup
   - Follow AAA pattern (Arrange, Act, Assert)
   - Mock external dependencies
   - Test both success and error cases

## Database Migrations

We use Alembic for database migrations:

```bash
# Create a new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one version
alembic downgrade -1
```

## Building and Packaging

1. **Development Build**

   ```bash
   python setup.py develop
   ```

2. **Production Build**

   ```bash
   python setup.py build
   ```

3. **Create Distribution**
   ```bash
   python setup.py sdist bdist_wheel
   ```

## Debugging

1. **Logging**

   - Logs are in `logs/spygate.log`
   - Set `LOG_LEVEL=DEBUG` in `.env` for detailed logs

2. **Debugging in VS Code**

   - Use the provided launch configurations in `.vscode/launch.json`
   - Set breakpoints in your code
   - Use the Debug Console for inspection

3. **Error Tracking**
   - Sentry integration for production error tracking
   - Local error logs for development

## Documentation

1. **Code Documentation**

   - Use Google-style docstrings
   - Document all public functions and classes
   - Include type hints

2. **Building Documentation**

   ```bash
   # Install documentation dependencies
   pip install -r docs/requirements.txt

   # Build documentation
   cd docs
   make html
   ```

## Git Workflow

1. **Branching Strategy**

   - `main` - production-ready code
   - `develop` - integration branch
   - `feature/*` - new features
   - `bugfix/*` - bug fixes
   - `release/*` - release preparation

2. **Commit Messages**
   Follow conventional commits:

   ```
   type(scope): description

   [optional body]

   [optional footer]
   ```

   Types: feat, fix, docs, style, refactor, test, chore

3. **Pull Requests**
   - Create PR against `develop`
   - Fill out PR template
   - Request review from team members
   - Ensure CI passes
   - Squash and merge

## Release Process

1. **Preparation**

   - Update version in `setup.py`
   - Update CHANGELOG.md
   - Create release branch

2. **Testing**

   - Run full test suite
   - Perform manual testing
   - Check documentation

3. **Release**
   - Merge to main
   - Tag release
   - Build and publish package
   - Deploy to production

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines on:

- How to submit changes
- Coding standards
- Commit message format
- Issue reporting
- Feature requests
