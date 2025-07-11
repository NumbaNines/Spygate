# Generated by Django 5.2.2 on 2025-06-08 12:52

import uuid

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="APIKey",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                ("key", models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ("name", models.CharField(max_length=100)),
                (
                    "permission_level",
                    models.CharField(
                        choices=[
                            ("read", "Read Only"),
                            ("write", "Read & Write"),
                            ("admin", "Admin Access"),
                        ],
                        default="read",
                        max_length=20,
                    ),
                ),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("expires_at", models.DateTimeField(blank=True, null=True)),
                ("last_used", models.DateTimeField(blank=True, null=True)),
                ("allowed_ips", models.JSONField(default=list)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="api_keys",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "API key",
                "verbose_name_plural": "API keys",
            },
        ),
        migrations.CreateModel(
            name="APIUsage",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                ("endpoint", models.CharField(max_length=255)),
                ("method", models.CharField(max_length=10)),
                ("status_code", models.IntegerField()),
                ("response_time", models.FloatField()),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
                ("ip_address", models.GenericIPAddressField()),
                ("user_agent", models.TextField()),
                ("request_data", models.JSONField(default=dict)),
                (
                    "api_key",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="usage",
                        to="api.apikey",
                    ),
                ),
            ],
            options={
                "verbose_name": "API usage",
                "verbose_name_plural": "API usages",
                "ordering": ["-timestamp"],
            },
        ),
        migrations.CreateModel(
            name="Webhook",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                ("name", models.CharField(max_length=100)),
                ("url", models.URLField()),
                ("secret", models.CharField(max_length=64)),
                ("event_types", models.JSONField(default=list)),
                ("is_active", models.BooleanField(default=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("last_triggered", models.DateTimeField(blank=True, null=True)),
                ("failure_count", models.IntegerField(default=0)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="webhooks",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
            options={
                "verbose_name": "webhook",
                "verbose_name_plural": "webhooks",
            },
        ),
        migrations.CreateModel(
            name="WebhookDelivery",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True, primary_key=True, serialize=False, verbose_name="ID"
                    ),
                ),
                ("event_type", models.CharField(max_length=50)),
                ("payload", models.JSONField()),
                ("response_status", models.IntegerField(null=True)),
                ("response_body", models.TextField(blank=True)),
                ("timestamp", models.DateTimeField(auto_now_add=True)),
                ("success", models.BooleanField(default=False)),
                ("retry_count", models.IntegerField(default=0)),
                (
                    "webhook",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="deliveries",
                        to="api.webhook",
                    ),
                ),
            ],
            options={
                "verbose_name": "webhook delivery",
                "verbose_name_plural": "webhook deliveries",
                "ordering": ["-timestamp"],
            },
        ),
    ]
