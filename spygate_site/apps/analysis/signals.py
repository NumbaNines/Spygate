from django.db.models.signals import post_save
from django.dispatch import receiver

from .models import GameAnalysis
from .tasks import analyze_video


@receiver(post_save, sender=GameAnalysis)
def handle_game_analysis_save(sender, instance, created, **kwargs):
    """
    Handle post-save signal for GameAnalysis model.
    """
    if created and instance.processing_status == "pending":
        analyze_video.delay(instance.id)
