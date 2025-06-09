# Django-YOLOv8 Integration Design Document

## Executive Summary

This document outlines the design and implementation approach for integrating the existing YOLOv8-based SpygateAI system with a Django web framework to create a hybrid desktop-web application architecture.

## Current State Analysis

### Existing YOLOv8 Implementation

- **Location**: `spygate/ml/yolo8_model.py` and `spygate/ml/yolov8_model.py`
- **Framework**: Ultralytics YOLOv8 with PyTorch backend
- **Features**:
  - Enhanced YOLOv8 with hardware-aware optimization
  - Advanced GPU memory management
  - Hardware tier-based model selection
  - UI element detection for HUD analysis
  - OCR integration with dual-engine processing

### System Architecture

```
Current: Desktop-Only (PyQt6)
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PyQt6 GUI     │───▶│  YOLOv8 Engine   │───▶│  Local Storage  │
│                 │    │  (ml/yolo8_model) │    │  (SQLite/Files) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

```
Proposed: Hybrid Desktop-Web
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PyQt6 GUI     │───▶│  YOLOv8 Engine   │───▶│  Local Storage  │
│   (Heavy Tasks) │    │  (Shared Core)    │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                           │
                              ▼                           ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  React Frontend │◀───│  Django REST API │───▶│  PostgreSQL DB  │
│  (Community)    │    │  (Web Backend)   │    │  (Cloud/Shared) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Design Principles

### 1. Separation of Concerns

- **Desktop App**: Heavy video processing, real-time analysis, local file management
- **Web Platform**: Community features, strategy sharing, lightweight analysis visualization
- **Shared Core**: YOLOv8 detection engine, data models, analysis algorithms

### 2. Performance Optimization

- Minimize video data transfer to web platform
- Use desktop app for processing-intensive YOLOv8 operations
- Web platform handles metadata, annotations, and social features

### 3. Progressive Enhancement

- Desktop app remains fully functional offline
- Web features enhance but don't replace core functionality
- Gradual migration of appropriate features to web

## Implementation Architecture

### Core Components

#### 1. Shared YOLOv8 Service Layer

```python
# services/yolo_service.py
from abc import ABC, abstractmethod
from typing import Union, Dict, Any
from ml.yolo8_model import EnhancedYOLOv8, DetectionResult

class YOLOServiceInterface(ABC):
    @abstractmethod
    def detect_hud_elements(self, frame: np.ndarray) -> DetectionResult:
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass

class SharedYOLOService(YOLOServiceInterface):
    """Shared YOLO service for both desktop and web platforms"""

    def __init__(self, config: Dict[str, Any] = None):
        self.model = EnhancedYOLOv8()
        self.config = config or {}

    def detect_hud_elements(self, frame: np.ndarray) -> DetectionResult:
        return self.model.detect_hud_elements(frame)

    def analyze_for_web(self, video_metadata: Dict) -> Dict[str, Any]:
        """Lightweight analysis for web upload"""
        # Process key frames only for web
        pass
```

#### 2. Django Models for YOLOv8 Data

```python
# models.py
from django.db import models
from django.contrib.auth.models import User

class DetectionSession(models.Model):
    """Represents a YOLOv8 detection session"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video_name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    hardware_tier = models.CharField(max_length=20)
    model_version = models.CharField(max_length=50)
    processing_time = models.FloatField()

class HUDDetection(models.Model):
    """Individual HUD element detection result"""
    session = models.ForeignKey(DetectionSession, on_delete=models.CASCADE)
    frame_number = models.IntegerField()
    element_type = models.CharField(max_length=50)  # UI_CLASSES
    confidence = models.FloatField()
    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()
    extracted_text = models.TextField(blank=True)

class GameSituation(models.Model):
    """Parsed game situation from HUD detections"""
    session = models.ForeignKey(DetectionSession, on_delete=models.CASCADE)
    frame_number = models.IntegerField()
    down = models.IntegerField(null=True, blank=True)
    distance = models.IntegerField(null=True, blank=True)
    field_position = models.CharField(max_length=10, blank=True)
    time_remaining = models.CharField(max_length=10, blank=True)
    home_score = models.IntegerField(null=True, blank=True)
    away_score = models.IntegerField(null=True, blank=True)
```

#### 3. Django REST API Endpoints

```python
# views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from celery import shared_task
from .services import SharedYOLOService

class YOLODetectionView(APIView):
    """API endpoint for YOLOv8 detection requests"""

    def post(self, request):
        # Handle lightweight detection requests from web
        pass

@shared_task
def process_video_with_yolo(video_path: str, user_id: int):
    """Celery task for background YOLOv8 processing"""
    service = SharedYOLOService()
    # Process video asynchronously
    pass

class DetectionResultsView(APIView):
    """Get detection results for web visualization"""

    def get(self, request, session_id):
        # Return detection results in web-friendly format
        pass
```

#### 4. Desktop-Web Bridge

```python
# desktop_bridge.py
import requests
from typing import Optional, Dict, Any

class DesktopWebBridge:
    """Bridge between desktop app and web platform"""

    def __init__(self, api_base_url: str, api_key: str):
        self.api_base_url = api_base_url
        self.api_key = api_key

    def upload_detection_results(self, results: DetectionResult, metadata: Dict) -> bool:
        """Upload detection results from desktop to web"""
        payload = {
            'detections': self._serialize_detections(results),
            'metadata': metadata
        }
        response = requests.post(
            f"{self.api_base_url}/api/detections/upload/",
            json=payload,
            headers={'Authorization': f'Bearer {self.api_key}'}
        )
        return response.status_code == 201

    def sync_user_preferences(self) -> Dict[str, Any]:
        """Sync user preferences from web to desktop"""
        pass
```

### Data Flow Patterns

#### 1. Desktop-First Processing

```
Video File → Desktop YOLOv8 → Local Storage → [Optional] Web Upload
```

#### 2. Web-Initiated Processing

```
Web Upload → Celery Task → Shared YOLOv8 → Database → Web Response
```

#### 3. Hybrid Analysis

```
Desktop Analysis → Bridge → Web API → Community Features
```

## Migration Strategy

### Phase 1: Foundation (Current Sprint)

- [x] Analyze existing YOLOv8 implementation
- [ ] Design shared service architecture
- [ ] Create Django models for detection data
- [ ] Implement basic REST API endpoints
- [ ] Build desktop-web bridge prototype

### Phase 2: Core Integration

- [ ] Refactor YOLOv8 service for shared usage
- [ ] Implement Celery task queue for web processing
- [ ] Create React components for detection visualization
- [ ] Build authentication and user management
- [ ] Implement basic data synchronization

### Phase 3: Advanced Features

- [ ] Real-time detection status via WebSocket
- [ ] Community sharing of detection strategies
- [ ] Cross-platform performance analytics
- [ ] Advanced visualization tools
- [ ] Mobile-responsive interface

### Phase 4: Optimization

- [ ] Performance tuning for web deployment
- [ ] Caching strategies for detection results
- [ ] CDN integration for global access
- [ ] A/B testing framework
- [ ] Production monitoring and alerts

## Technical Considerations

### Hardware Compatibility

- **Desktop**: Full YOLOv8 with hardware optimization
- **Web Server**: CPU-optimized models for broader compatibility
- **Edge Cases**: Graceful degradation for limited hardware

### Performance Targets

- **Desktop Detection**: Maintain current 0.038s-1.170s performance
- **Web Processing**: <30s for typical clip analysis
- **API Response**: <200ms for metadata requests
- **Real-time Updates**: <1s latency via WebSocket

### Security

- **API Authentication**: JWT tokens with refresh mechanism
- **Data Encryption**: TLS 1.3 for all communications
- **Video Privacy**: Optional local-only processing mode
- **Rate Limiting**: Prevent API abuse

### Scalability

- **Horizontal Scaling**: Celery workers for processing load
- **Database Optimization**: Indexed queries for large datasets
- **Caching**: Redis for frequently accessed detection results
- **CDN**: Static assets and visualization data

## Implementation Timeline

### Week 1-2: Foundation

- Shared service architecture design
- Django models and basic API
- Desktop bridge prototype

### Week 3-4: Core Features

- YOLOv8 service refactoring
- Celery integration
- Basic React components

### Week 5-6: Integration Testing

- End-to-end workflow testing
- Performance benchmarking
- Bug fixes and optimization

### Week 7-8: Polish & Deploy

- Production deployment setup
- Documentation completion
- User acceptance testing

## Success Metrics

### Technical Metrics

- **Detection Accuracy**: Maintain ≥95% of desktop performance
- **API Latency**: <200ms average response time
- **Processing Speed**: Complete clip analysis within 30 seconds
- **System Uptime**: 99.5% availability

### User Metrics

- **Desktop-Web Sync**: 100% successful synchronization rate
- **Feature Adoption**: 80% of users engage with web features
- **Performance Satisfaction**: >4.5/5 user rating
- **Community Engagement**: Measurable increase in strategy sharing

## Risk Mitigation

### Technical Risks

- **Performance Degradation**: Maintain desktop app as primary processor
- **Model Compatibility**: Version control for YOLOv8 models
- **Data Loss**: Comprehensive backup and sync strategies
- **Security Vulnerabilities**: Regular security audits

### Business Risks

- **User Adoption**: Gradual feature rollout with feedback loops
- **Resource Costs**: Efficient processing to minimize server costs
- **Competition**: Focus on unique community features
- **Maintenance Burden**: Automated testing and deployment

## Conclusion

This design provides a robust foundation for integrating YOLOv8 with Django while maintaining the performance and reliability of the existing desktop application. The hybrid architecture enables new community features while preserving the core strengths of the current implementation.

The phased approach ensures minimal disruption to existing users while progressively adding value through web-based collaboration features. Success depends on careful performance monitoring and maintaining the desktop app's processing advantages.
