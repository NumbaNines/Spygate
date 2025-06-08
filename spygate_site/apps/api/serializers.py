from rest_framework import serializers
from apps.downloads.models import Release, Download
from apps.accounts.models import User, UserProfile
from .models import GameAnalysis, Formation, Play, Situation

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'first_name', 'last_name', 'is_verified')
        read_only_fields = ('id', 'is_verified')

class UserProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = UserProfile
        fields = (
            'user', 'subscription_type', 'download_count',
            'last_download', 'storage_quota', 'storage_used',
            'notify_updates', 'notify_releases', 'notify_security'
        )
        read_only_fields = ('download_count', 'storage_used')

class ReleaseSerializer(serializers.ModelSerializer):
    download_url = serializers.SerializerMethodField()
    analyze_url = serializers.SerializerMethodField()
    
    class Meta:
        model = Release
        fields = (
            'id', 'version', 'version_type', 'release_notes',
            'changelog', 'checksum', 'file_size', 'download_url',
            'analyze_url', 'is_active', 'is_critical', 'release_date'
        )
        read_only_fields = ('checksum', 'file_size', 'download_url', 'analyze_url')
    
    def get_download_url(self, obj):
        request = self.context.get('request')
        if request is None:
            return None
        return request.build_absolute_uri(f'/api/releases/{obj.id}/download/')
    
    def get_analyze_url(self, obj):
        request = self.context.get('request')
        if request is None:
            return None
        return request.build_absolute_uri(f'/api/releases/{obj.id}/analyze/')

class DownloadSerializer(serializers.ModelSerializer):
    release = ReleaseSerializer(read_only=True)
    user = UserSerializer(read_only=True)
    
    class Meta:
        model = Download
        fields = ('id', 'user', 'release', 'download_date', 'file_size', 'status')
        read_only_fields = fields 

class GameAnalysisSerializer(serializers.ModelSerializer):
    """Serializer for game analysis results."""
    class Meta:
        model = GameAnalysis
        fields = '__all__'
        read_only_fields = ('user', 'started_at', 'completed_at', 'processing_status', 'error_message')

class FormationSerializer(serializers.ModelSerializer):
    """Serializer for detected formations."""
    class Meta:
        model = Formation
        fields = '__all__'
        read_only_fields = ('analysis',)

class PlaySerializer(serializers.ModelSerializer):
    """Serializer for detected plays."""
    formation = FormationSerializer(read_only=True)
    
    class Meta:
        model = Play
        fields = '__all__'
        read_only_fields = ('analysis',)

class SituationSerializer(serializers.ModelSerializer):
    """Serializer for game situations."""
    class Meta:
        model = Situation
        fields = '__all__'
        read_only_fields = ('analysis',) 