from django.urls import include, path
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView, TokenVerifyView

from . import views

router = DefaultRouter()
router.register(r"analysis", views.GameAnalysisViewSet, basename="analysis")
router.register(r"formations", views.FormationViewSet, basename="formation")
router.register(r"plays", views.PlayViewSet, basename="play")
router.register(r"situations", views.SituationViewSet, basename="situation")
router.register(r"metrics", views.UserMetricsViewSet, basename="metrics")
router.register(r"feature-usage", views.FeatureUsageViewSet, basename="feature-usage")
router.register(r"performance", views.PerformanceMetricViewSet, basename="performance")

urlpatterns = [
    path("", include(router.urls)),
    path("token/", TokenObtainPairView.as_view(), name="token_obtain_pair"),
    path("token/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("token/verify/", TokenVerifyView.as_view(), name="token_verify"),
]
