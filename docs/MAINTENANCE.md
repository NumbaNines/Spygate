# SpygateAI Maintenance Guide

This guide provides detailed instructions for managing and maintaining the SpygateAI project.

## Project Size Management

### Storage Guidelines

- Keep project size under 1GB for optimal performance
- Use Git LFS for files larger than 100MB
- Store only one sample test video in `tests/test_videos/`
- Regular cleanup using `cleanup.ps1` script

### Running Cleanup Script

```powershell
# From project root
.\cleanup.ps1
```

The cleanup script handles:

- Removing virtual environments
- Clearing cache files
- Cleaning pip cache
- Removing **pycache** directories
- Backing up critical files

## Virtual Environment Management

### Creating New Environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Updating Dependencies

1. Update requirements:
   ```bash
   pip freeze > requirements.txt
   ```
2. Review and update version numbers
3. Test compatibility:
   ```bash
   pip install -r requirements.txt
   pytest
   ```

## Git Management

### Large File Storage (LFS)

- Use Git LFS for:
  - Model files (_.pt, _.pth)
  - Video files (\*.mp4)
  - Large datasets (\*.csv over 50MB)
  - Binary files (_.dll, _.so)

### Repository Maintenance

1. Regular garbage collection:
   ```bash
   git gc --aggressive
   ```
2. Clean old branches:
   ```bash
   git remote prune origin
   ```
3. Update LFS objects:
   ```bash
   git lfs prune
   ```

## Testing and Quality Assurance

### Running Tests

```bash
# Full test suite
pytest

# With coverage
pytest --cov=src

# Generate coverage report
coverage html
```

### Code Quality Checks

1. Style formatting:
   ```bash
   black src tests
   ```
2. Type checking:
   ```bash
   mypy src
   ```
3. Linting:
   ```bash
   flake8 src tests
   ```

## Documentation Management

### Updating Documentation

1. Update relevant .md files in `/docs`
2. For API changes, update:
   - API reference docs
   - Example code
   - Docstrings

### Building Documentation

```bash
cd docs
make html  # Builds HTML documentation
```

## Backup Procedures

### Critical Files to Backup

- Configuration files
- Custom models
- Test data
- User settings

### Backup Process

1. Use built-in backup in cleanup script:
   ```powershell
   .\cleanup.ps1 -CreateBackup
   ```
2. Regular backups stored in `backups/` directory

## Performance Monitoring

### System Requirements

- Python 3.9+
- Windows OS
- NVIDIA GPU (recommended)
- Minimum 8GB RAM

### Monitoring Tools

- Use Windows Task Manager for resource usage
- Python profiling:
  ```bash
  python -m cProfile -o profile.stats your_script.py
  ```

## Troubleshooting Common Issues

### Virtual Environment Problems

1. Delete existing environment:
   ```bash
   rm -rf .venv
   ```
2. Create fresh environment
3. Reinstall dependencies

### Git LFS Issues

1. Verify LFS installation:
   ```bash
   git lfs install
   ```
2. Re-download LFS files:
   ```bash
   git lfs pull
   ```

### Package Conflicts

1. Generate dependency tree:
   ```bash
   pip install pipdeptree
   pipdeptree
   ```
2. Resolve conflicts in requirements.txt

## Regular Maintenance Schedule

### Daily

- Push code changes
- Update documentation for new features
- Run tests for modified code

### Weekly

- Run full test suite
- Update dependencies if needed
- Clean up temporary files

### Monthly

- Full system cleanup
- Git repository maintenance
- Update documentation
- Review and update requirements.txt

## Contact and Support

For issues or questions:

1. Check existing documentation
2. Search closed GitHub issues
3. Open new issue with detailed description
4. Contact maintainers through Discord

Remember to always test changes in a development environment before applying to production.
