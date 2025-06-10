#!/usr/bin/env python3
"""
SpygateAI User Database Management
==================================

Handles user accounts, authentication, and premium subscriptions.
"""

import hashlib
import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class User:
    """User data model"""

    user_id: int
    username: str
    display_name: str
    email: str
    profile_picture: str
    profile_picture_type: str  # "emoji" or "custom"
    created_at: str
    last_login: str
    is_premium: bool = False
    subscription_type: str = "free"
    subscription_expires: Optional[str] = None


@dataclass
class Subscription:
    """Subscription data model"""

    subscription_id: int
    user_id: int
    plan_type: str  # "free", "pro", "premium"
    status: str  # "active", "cancelled", "expired"
    started_at: str
    expires_at: Optional[str]
    price_paid: float
    features: str  # JSON string of features


class UserDatabase:
    """Manages user database operations"""

    def __init__(self, db_path: str = "spygate_users.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Users table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    display_name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    profile_picture TEXT DEFAULT '🏈',
                    profile_picture_type TEXT DEFAULT 'emoji',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_premium BOOLEAN DEFAULT 0,
                    subscription_type TEXT DEFAULT 'free',
                    subscription_expires TIMESTAMP NULL
                )
            """
            )

            # Subscriptions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS subscriptions (
                    subscription_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    plan_type TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NULL,
                    price_paid REAL DEFAULT 0.0,
                    features TEXT DEFAULT '{}',
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """
            )

            # User settings table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS user_settings (
                    setting_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    setting_key TEXT NOT NULL,
                    setting_value TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    UNIQUE(user_id, setting_key)
                )
            """
            )

            conn.commit()

    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = "spygate_salt_2024"  # In production, use random salt per user
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def create_user(
        self,
        username: str,
        display_name: str,
        email: str,
        password: str,
        profile_picture: str = "🏈",
        profile_picture_type: str = "emoji",
    ) -> Optional[User]:
        """Create a new user"""
        password_hash = self.hash_password(password)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO users (username, display_name, email, password_hash, profile_picture, profile_picture_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        username,
                        display_name,
                        email,
                        password_hash,
                        profile_picture,
                        profile_picture_type,
                    ),
                )

                user_id = cursor.lastrowid
                conn.commit()

                # Create default free subscription
                self.create_subscription(user_id, "free", 0.0)

                return self.get_user_by_id(user_id)

            except sqlite3.IntegrityError:
                return None  # User already exists

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, username, display_name, email, profile_picture, profile_picture_type,
                       created_at, last_login, is_premium, subscription_type, subscription_expires
                FROM users WHERE username = ?
            """,
                (username,),
            )

            row = cursor.fetchone()
            if row:
                return User(*row)
            return None

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT user_id, username, display_name, email, profile_picture, profile_picture_type,
                       created_at, last_login, is_premium, subscription_type, subscription_expires
                FROM users WHERE user_id = ?
            """,
                (user_id,),
            )

            row = cursor.fetchone()
            if row:
                return User(*row)
            return None

    def update_user_profile_picture(
        self, user_id: int, profile_picture: str, profile_picture_type: str = "emoji"
    ) -> bool:
        """Update user's profile picture"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users SET profile_picture = ?, profile_picture_type = ? WHERE user_id = ?
            """,
                (profile_picture, profile_picture_type, user_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_last_login(self, user_id: int):
        """Update user's last login timestamp"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?
            """,
                (user_id,),
            )
            conn.commit()

    def create_subscription(
        self, user_id: int, plan_type: str, price_paid: float, duration_months: int = 0
    ) -> int:
        """Create a new subscription"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Calculate expiration date
            expires_at = None
            if duration_months > 0:
                expires_at = (datetime.now() + timedelta(days=duration_months * 30)).isoformat()

            # Define features based on plan type
            features = self.get_plan_features(plan_type)

            cursor.execute(
                """
                INSERT INTO subscriptions (user_id, plan_type, expires_at, price_paid, features)
                VALUES (?, ?, ?, ?, ?)
            """,
                (user_id, plan_type, expires_at, price_paid, json.dumps(features)),
            )

            subscription_id = cursor.lastrowid

            # Update user's subscription info
            is_premium = 1 if plan_type != "free" else 0
            cursor.execute(
                """
                UPDATE users SET is_premium = ?, subscription_type = ?, subscription_expires = ?
                WHERE user_id = ?
            """,
                (is_premium, plan_type, expires_at, user_id),
            )

            conn.commit()
            return subscription_id

    def get_plan_features(self, plan_type: str) -> dict[str, Any]:
        """Get features for a subscription plan"""
        features = {
            "free": {
                "video_analysis_limit": 3,
                "formation_analysis": True,
                "basic_stats": True,
                "export_clips": False,
                "advanced_analytics": False,
                "custom_formations": False,
                "ai_coaching": False,
            },
            "pro": {
                "video_analysis_limit": 25,
                "formation_analysis": True,
                "basic_stats": True,
                "export_clips": True,
                "advanced_analytics": True,
                "custom_formations": True,
                "ai_coaching": False,
            },
            "premium": {
                "video_analysis_limit": -1,  # Unlimited
                "formation_analysis": True,
                "basic_stats": True,
                "export_clips": True,
                "advanced_analytics": True,
                "custom_formations": True,
                "ai_coaching": True,
                "priority_support": True,
                "beta_features": True,
            },
        }
        return features.get(plan_type, features["free"])

    def get_user_subscriptions(self, user_id: int) -> list[Subscription]:
        """Get all subscriptions for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT subscription_id, user_id, plan_type, status, started_at,
                       expires_at, price_paid, features
                FROM subscriptions WHERE user_id = ?
                ORDER BY started_at DESC
            """,
                (user_id,),
            )

            return [Subscription(*row) for row in cursor.fetchall()]

    def check_subscription_status(self, user_id: int) -> dict[str, Any]:
        """Check current subscription status"""
        user = self.get_user_by_id(user_id)
        if not user:
            return {"status": "not_found"}

        if not user.is_premium:
            return {"status": "free", "plan": "free"}

        # Check if subscription has expired
        if user.subscription_expires:
            expires_at = datetime.fromisoformat(
                user.subscription_expires.replace("Z", "+00:00").replace("+00:00", "")
            )
            if datetime.now() > expires_at:
                # Subscription expired, downgrade to free
                self.downgrade_to_free(user_id)
                return {"status": "expired", "plan": "free"}

        return {
            "status": "active",
            "plan": user.subscription_type,
            "expires_at": user.subscription_expires,
        }

    def downgrade_to_free(self, user_id: int):
        """Downgrade user to free plan"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE users SET is_premium = 0, subscription_type = 'free',
                                subscription_expires = NULL WHERE user_id = ?
            """,
                (user_id,),
            )
            conn.commit()

    def get_user_setting(self, user_id: int, setting_key: str) -> Optional[str]:
        """Get a user setting"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT setting_value FROM user_settings
                WHERE user_id = ? AND setting_key = ?
            """,
                (user_id, setting_key),
            )

            row = cursor.fetchone()
            return row[0] if row else None

    def set_user_setting(self, user_id: int, setting_key: str, setting_value: str):
        """Set a user setting"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO user_settings (user_id, setting_key, setting_value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (user_id, setting_key, setting_value),
            )
            conn.commit()


def setup_demo_user():
    """Setup demo user NumbaNines with premium subscription"""
    db = UserDatabase()

    # Create the demo user with football emoji as default
    user = db.create_user(
        username="NumbaNines",
        display_name="NumbaNines",
        email="numbanines@spygate.ai",
        password="demo123",
        profile_picture="🏈",  # Default to football emoji
        profile_picture_type="emoji",
    )

    if user:
        # Give them a premium subscription (18 months)
        subscription_id = db.create_subscription(user.user_id, "premium", 19.99, 18)
        print(f"✅ Created demo user: {user.display_name}")
        print(f"✅ Premium subscription ID: {subscription_id}")

        # Update to trophy emoji for demo
        db.update_user_profile_picture(user.user_id, "🏆", "emoji")
        return db.get_user_by_id(user.user_id)

    return None


if __name__ == "__main__":
    # Initialize database and create demo user
    user = setup_demo_user()
    if user:
        print(f"User created: {user.display_name} ({user.subscription_type})")
