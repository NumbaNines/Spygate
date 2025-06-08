from django.db import models
from django.utils.translation import gettext_lazy as _
from apps.accounts.models import User
from apps.downloads.models import Release

class GameAnalysis(models.Model):
    """Model for storing game analysis results."""
    PROCESSING_STATUS = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video_file = models.FileField(upload_to='game_videos/')
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processing_status = models.CharField(max_length=20, choices=PROCESSING_STATUS, default='pending')
    completed_at = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)
    metadata = models.JSONField(default=dict)

    class Meta:
        verbose_name = _('game analysis')
        verbose_name_plural = _('game analyses')
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} - {self.user.username}"

class Formation(models.Model):
    """Model for storing formation data."""
    FORMATION_TYPES = (
        ('offense', 'Offense'),
        ('defense', 'Defense'),
    )

    analysis = models.ForeignKey(GameAnalysis, on_delete=models.CASCADE, related_name='formations')
    name = models.CharField(max_length=100)
    formation_type = models.CharField(max_length=20, choices=FORMATION_TYPES)
    timestamp = models.FloatField()  # Timestamp in video where formation was detected
    confidence_score = models.FloatField()
    player_positions = models.JSONField()  # Store player positions as JSON
    metadata = models.JSONField(default=dict)

    class Meta:
        verbose_name = _('formation')
        verbose_name_plural = _('formations')
        ordering = ['timestamp']

    def __str__(self):
        return f"{self.name} at {self.timestamp}s"

class Play(models.Model):
    """Model for storing play data."""
    PLAY_TYPES = (
        ('run', 'Run'),
        ('pass', 'Pass'),
        ('special', 'Special Teams'),
    )

    analysis = models.ForeignKey(GameAnalysis, on_delete=models.CASCADE, related_name='plays')
    formation = models.ForeignKey(Formation, on_delete=models.SET_NULL, null=True, related_name='plays')
    name = models.CharField(max_length=100)
    play_type = models.CharField(max_length=20, choices=PLAY_TYPES)
    start_time = models.FloatField()
    end_time = models.FloatField()
    success_rate = models.FloatField(null=True)
    yards_gained = models.IntegerField(null=True)
    player_routes = models.JSONField()  # Store player routes as JSON
    metadata = models.JSONField(default=dict)

    class Meta:
        verbose_name = _('play')
        verbose_name_plural = _('plays')
        ordering = ['start_time']

    def __str__(self):
        return f"{self.name} ({self.start_time}s - {self.end_time}s)"

class Situation(models.Model):
    """Model for storing game situation data."""
    analysis = models.ForeignKey(GameAnalysis, on_delete=models.CASCADE, related_name='situations')
    down = models.IntegerField()
    distance = models.IntegerField()
    field_position = models.IntegerField()  # Yard line
    score_differential = models.IntegerField()
    time_remaining = models.IntegerField()  # Seconds remaining
    quarter = models.IntegerField()
    is_red_zone = models.BooleanField(default=False)
    metadata = models.JSONField(default=dict)

    class Meta:
        verbose_name = _('situation')
        verbose_name_plural = _('situations')
        ordering = ['time_remaining']

    def __str__(self):
        return f"{self.down} & {self.distance} at {self.field_position}"

class UserMetrics(models.Model):
    """Model for tracking user-specific metrics."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='metrics')
    last_active = models.DateTimeField(auto_now=True)
    total_downloads = models.IntegerField(default=0)
    total_storage_used = models.BigIntegerField(default=0)  # in bytes
    favorite_features = models.JSONField(default=list)
    system_info = models.JSONField(default=dict)
    
    class Meta:
        verbose_name = _('user metrics')
        verbose_name_plural = _('user metrics')

    def __str__(self):
        return f"Metrics for {self.user.username}"

class FeatureUsage(models.Model):
    """Model for tracking feature usage."""
    FEATURE_CATEGORIES = (
        ('core', 'Core Features'),
        ('analysis', 'Analysis Tools'),
        ('export', 'Export Features'),
        ('integration', 'Integrations'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    feature_name = models.CharField(max_length=100)
    feature_category = models.CharField(max_length=20, choices=FEATURE_CATEGORIES)
    usage_count = models.IntegerField(default=0)
    last_used = models.DateTimeField(auto_now=True)
    success_rate = models.FloatField(default=0.0)
    average_duration = models.DurationField(null=True)
    
    class Meta:
        verbose_name = _('feature usage')
        verbose_name_plural = _('feature usages')
        unique_together = ['user', 'feature_name']

    def __str__(self):
        return f"{self.user.username} - {self.feature_name}"

class ErrorReport(models.Model):
    """Model for tracking application errors."""
    ERROR_LEVELS = (
        ('debug', 'Debug'),
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('error', 'Error'),
        ('critical', 'Critical'),
    )

    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    release = models.ForeignKey(Release, on_delete=models.SET_NULL, null=True)
    error_level = models.CharField(max_length=20, choices=ERROR_LEVELS)
    error_message = models.TextField()
    stack_trace = models.TextField(blank=True)
    system_info = models.JSONField(default=dict)
    timestamp = models.DateTimeField(auto_now_add=True)
    resolved = models.BooleanField(default=False)
    resolution_notes = models.TextField(blank=True)
    
    class Meta:
        verbose_name = _('error report')
        verbose_name_plural = _('error reports')
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.error_level} - {self.timestamp}"

class PerformanceMetric(models.Model):
    """Model for tracking application performance metrics."""
    METRIC_TYPES = (
        ('cpu', 'CPU Usage'),
        ('memory', 'Memory Usage'),
        ('latency', 'Network Latency'),
        ('fps', 'Frames Per Second'),
        ('load_time', 'Load Time'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    metric_type = models.CharField(max_length=20, choices=METRIC_TYPES)
    value = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    system_info = models.JSONField(default=dict)
    
    class Meta:
        verbose_name = _('performance metric')
        verbose_name_plural = _('performance metrics')
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.metric_type} - {self.timestamp}"

class UsageSession(models.Model):
    """Model for tracking user sessions."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='sessions')
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    duration = models.DurationField(null=True, blank=True)
    ip_address = models.GenericIPAddressField()
    user_agent = models.TextField()
    features_used = models.JSONField(default=list)
    session_data = models.JSONField(default=dict)
    
    class Meta:
        verbose_name = _('usage session')
        verbose_name_plural = _('usage sessions')
        ordering = ['-start_time']

    def __str__(self):
        return f"{self.user.username} - {self.start_time}"

    def calculate_duration(self):
        """Calculate session duration if end time is set."""
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time
            return self.duration
        return None
