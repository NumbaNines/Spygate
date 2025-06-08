# Getting Started with Spygate

This guide will help you get Spygate up and running on your system.

## Prerequisites

- Python 3.10 or higher
- Git (for development)
- FFmpeg (for video processing)
- Qt6 (installed automatically with dependencies)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/spygate.git
   cd spygate
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix/MacOS:
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Copy example environment file
   cp .env.example .env
   # Edit .env with your settings
   ```

## Configuration

1. Database Setup:

   - The SQLite database will be automatically created on first run
   - Default location: `data/spygate.db`
   - No additional configuration needed

2. Error Tracking (Optional):

   - Sign up for a free Sentry account at https://sentry.io
   - Add your Sentry DSN to the `.env` file:
     ```
     SENTRY_DSN=your_dsn_here
     ```

3. Logging:
   - Logs are stored in the `logs` directory
   - Default log level is INFO
   - Configure log level in `.env`:
     ```
     LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR
     ```

## Quick Start

1. Start the application:

   ```bash
   python -m spygate
   ```

2. Import a video:

   - Click "Import Video" in the main window
   - Select a Madden NFL 25 game recording
   - Enter player information when prompted

3. Create clips:

   - Use the video controls to navigate
   - Click "Create Clip" at interesting moments
   - Add tags and notes to your clips

4. Analyze plays:
   - Select a clip from the list
   - Use the analysis tools to break down the play
   - Save your analysis

## Next Steps

- Read the [User Guide](./user-guide/README.md) for detailed usage instructions
- Check out [Example Workflows](./user-guide/workflows.md) for common use cases
- Join our [Community](./community.md) for support and discussions

## Troubleshooting

If you encounter any issues during installation or setup:

1. Check the [Troubleshooting Guide](./troubleshooting.md)
2. Verify all prerequisites are installed correctly
3. Ensure your Python version is compatible
4. Check the application logs in the `logs` directory

For additional help, please [create an issue](https://github.com/yourusername/spygate/issues) on GitHub.
