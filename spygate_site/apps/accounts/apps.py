from django.apps import AppConfig


class AccountsConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.accounts'
    verbose_name = 'User Accounts'

    def ready(self):
        """Initialize app when Django starts."""
        try:
            import apps.accounts.signals  # noqa
        except ImportError:
            pass  # Handle gracefully if signals module doesn't exist yet
