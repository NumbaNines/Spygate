from django.contrib.auth.models import AbstractUser
from django.db import models
from django.utils.translation import gettext_lazy as _

class User(AbstractUser):
    """Custom user model for SpygateAI."""
    email = models.EmailField(_('email address'), unique=True)
    company_name = models.CharField(max_length=255, blank=True)
    is_verified = models.BooleanField(default=False)
    
    # Additional fields for user tracking
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    registration_ip = models.GenericIPAddressField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = _('user')
        verbose_name_plural = _('users')

class UserProfile(models.Model):
    """Extended profile information for users."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    subscription_type = models.CharField(max_length=50, default='free')
    download_count = models.IntegerField(default=0)
    last_download = models.DateTimeField(null=True, blank=True)
    storage_quota = models.BigIntegerField(default=5242880)  # 5MB in bytes
    storage_used = models.BigIntegerField(default=0)
    
    # Notification preferences
    notify_updates = models.BooleanField(default=True)
    notify_releases = models.BooleanField(default=True)
    notify_security = models.BooleanField(default=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = _('user profile')
        verbose_name_plural = _('user profiles')

    def __str__(self):
        return f"{self.user.username}'s profile"

    def can_download(self):
        """Check if user can download based on subscription type."""
        if self.subscription_type == 'free':
            return self.download_count < 3  # Free users get 3 downloads
        return True

    def has_storage_space(self, file_size):
        """Check if user has enough storage space."""
        return (self.storage_used + file_size) <= self.storage_quota

class LoginHistory(models.Model):
    """Track user login history."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='login_history')
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField()
    success = models.BooleanField(default=True)
    
    class Meta:
        verbose_name = _('login history')
        verbose_name_plural = _('login histories')
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user.username} - {self.timestamp}"
