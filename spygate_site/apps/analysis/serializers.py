from rest_framework import serializers
from .models import GameAnalysis, Formation, Play, Situation, UserMetrics, FeatureUsage, PerformanceMetric

class SituationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Situation
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')

class FormationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Formation
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')

class PlaySerializer(serializers.ModelSerializer):
    formation = FormationSerializer(read_only=True)
    situation = SituationSerializer(read_only=True)

    class Meta:
        model = Play
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')

class GameAnalysisSerializer(serializers.ModelSerializer):
    formations = FormationSerializer(many=True, read_only=True)
    plays = PlaySerializer(many=True, read_only=True)
    situations = SituationSerializer(many=True, read_only=True)
    
    class Meta:
        model = GameAnalysis
        fields = '__all__'
        read_only_fields = ('user', 'created_at', 'updated_at', 'processing_status', 'completed_at')

    def validate_video_file(self, value):
        """Validate video file size and format."""
        if value.size > 1024 * 1024 * 500:  # 500MB limit
            raise serializers.ValidationError("Video file size cannot exceed 500MB.")
        
        # Check file extension
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        ext = value.name.lower().split('.')[-1]
        if f'.{ext}' not in allowed_extensions:
            raise serializers.ValidationError(f"Invalid file format. Allowed formats: {', '.join(allowed_extensions)}")
        
        return value

class UserMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserMetrics
        fields = '__all__'
        read_only_fields = ('user', 'last_active', 'total_downloads', 'total_storage_used')

class FeatureUsageSerializer(serializers.ModelSerializer):
    class Meta:
        model = FeatureUsage
        fields = '__all__'
        read_only_fields = ('user', 'usage_count', 'last_used', 'success_rate', 'average_duration')

class PerformanceMetricSerializer(serializers.ModelSerializer):
    class Meta:
        model = PerformanceMetric
        fields = '__all__'
        read_only_fields = ('user', 'timestamp') 