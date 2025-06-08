import uuid

from apps.accounts.models import User
from django.db import models
from django.utils.translation import gettext_lazy as _


class APIKey(models.Model):
    """Model for managing API keys."""

    PERMISSION_LEVELS = (
        ("read", "Read Only"),
        ("write", "Read & Write"),
        ("admin", "Admin Access"),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="api_keys")
    key = models.UUIDField(default=uuid.uuid4, editable=False, unique=True)
    name = models.CharField(max_length=100)
    permission_level = models.CharField(max_length=20, choices=PERMISSION_LEVELS, default="read")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    last_used = models.DateTimeField(null=True, blank=True)
    allowed_ips = models.JSONField(default=list)  # List of allowed IP addresses

    class Meta:
        verbose_name = _("API key")
        verbose_name_plural = _("API keys")

    def __str__(self):
        return f"{self.name} - {self.user.username}"

    @property
    def is_valid(self):
        """Check if API key is valid."""
        from django.utils import timezone

        if not self.is_active:
            return False
        if self.expires_at and self.expires_at < timezone.now():
            return False
        return True


class APIUsage(models.Model):
    """Model for tracking API usage."""

    api_key = models.ForeignKey(APIKey, on_delete=models.CASCADE, related_name="usage")
    endpoint = models.CharField(max_length=255)
    method = models.CharField(max_length=10)  # GET, POST, etc.
    status_code = models.IntegerField()
    response_time = models.FloatField()  # in seconds
    timestamp = models.DateTimeField(auto_now_add=True)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField()
    request_data = models.JSONField(default=dict)

    class Meta:
        verbose_name = _("API usage")
        verbose_name_plural = _("API usages")
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.api_key.name} - {self.endpoint} - {self.timestamp}"


class Webhook(models.Model):
    """Model for managing webhooks."""

    EVENT_TYPES = (
        ("download", "Download Events"),
        ("release", "Release Events"),
        ("error", "Error Events"),
        ("user", "User Events"),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="webhooks")
    name = models.CharField(max_length=100)
    url = models.URLField()
    secret = models.CharField(max_length=64)  # For signature verification
    event_types = models.JSONField(default=list)  # List of event types to trigger webhook
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_triggered = models.DateTimeField(null=True, blank=True)
    failure_count = models.IntegerField(default=0)

    class Meta:
        verbose_name = _("webhook")
        verbose_name_plural = _("webhooks")

    def __str__(self):
        return f"{self.name} - {self.user.username}"


class WebhookDelivery(models.Model):
    """Model for tracking webhook deliveries."""

    webhook = models.ForeignKey(Webhook, on_delete=models.CASCADE, related_name="deliveries")
    event_type = models.CharField(max_length=50)
    payload = models.JSONField()
    response_status = models.IntegerField(null=True)
    response_body = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    success = models.BooleanField(default=False)
    retry_count = models.IntegerField(default=0)

    class Meta:
        verbose_name = _("webhook delivery")
        verbose_name_plural = _("webhook deliveries")
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.webhook.name} - {self.event_type} - {self.timestamp}"


class GameAnalysis(models.Model):
    """Model for storing game analysis results."""

    GAME_VERSIONS = (
        ("madden_25", "Madden NFL 25"),
        ("cfb_25", "College Football 25"),
    )

    HARDWARE_TIERS = (
        ("minimum", "Minimum"),
        ("standard", "Standard"),
        ("premium", "Premium"),
        ("professional", "Professional"),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="analyses")
    video_file = models.FileField(upload_to="analysis_videos/")
    game_version = models.CharField(max_length=20, choices=GAME_VERSIONS)
    hardware_tier = models.CharField(max_length=20, choices=HARDWARE_TIERS)
    processing_status = models.CharField(max_length=20, default="pending")
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    class Meta:
        verbose_name = _("game analysis")
        verbose_name_plural = _("game analyses")
        ordering = ["-started_at"]

    def __str__(self):
        return f"{self.user.username} - {self.game_version} - {self.started_at}"


class Formation(models.Model):
    """Model for storing detected formations."""

    analysis = models.ForeignKey(GameAnalysis, on_delete=models.CASCADE, related_name="formations")
    name = models.CharField(max_length=100)
    timestamp = models.FloatField()  # Time in video where formation was detected
    confidence = models.FloatField()  # Detection confidence score
    player_positions = models.JSONField()  # JSON array of player positions
    metadata = models.JSONField(default=dict)  # Additional formation data

    class Meta:
        verbose_name = _("formation")
        verbose_name_plural = _("formations")
        ordering = ["timestamp"]

    def __str__(self):
        return f"{self.name} at {self.timestamp}s"


class Play(models.Model):
    """Model for storing detected plays."""

    analysis = models.ForeignKey(GameAnalysis, on_delete=models.CASCADE, related_name="plays")
    formation = models.ForeignKey(
        Formation, on_delete=models.SET_NULL, null=True, related_name="plays"
    )
    name = models.CharField(max_length=100)
    play_type = models.CharField(max_length=50)  # run, pass, special teams
    timestamp = models.FloatField()
    confidence = models.FloatField()
    result = models.JSONField()  # Play outcome data
    metadata = models.JSONField(default=dict)

    class Meta:
        verbose_name = _("play")
        verbose_name_plural = _("plays")
        ordering = ["timestamp"]

    def __str__(self):
        return f"{self.name} ({self.play_type}) at {self.timestamp}s"


class Situation(models.Model):
    """Model for storing game situations."""

    analysis = models.ForeignKey(GameAnalysis, on_delete=models.CASCADE, related_name="situations")
    timestamp = models.FloatField()
    down = models.IntegerField()
    distance = models.IntegerField()
    field_position = models.IntegerField()  # Yard line
    score_home = models.IntegerField()
    score_away = models.IntegerField()
    quarter = models.IntegerField()
    time_remaining = models.IntegerField()  # Seconds remaining in quarter
    metadata = models.JSONField(default=dict)

    class Meta:
        verbose_name = _("situation")
        verbose_name_plural = _("situations")
        ordering = ["timestamp"]

    def __str__(self):
        return f"{self.down} & {self.distance} at {self.timestamp}s"
