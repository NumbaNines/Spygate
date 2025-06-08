from django.apps import AppConfig


class AnalysisConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.analysis'
    verbose_name = 'Game Analysis'
    
    def ready(self):
        """
        Initialize app when Django starts.
        """
        try:
            import apps.analysis.signals  # noqa
        except ImportError:
            pass  # Handle gracefully if signals module doesn't exist yet
