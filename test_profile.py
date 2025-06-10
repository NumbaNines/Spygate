#!/usr/bin/env python3
"""
Test Profile Picture System
===========================
"""

from profile_picture_manager import ProfilePictureManager, is_emoji_profile
from user_database import UserDatabase


def test_profile_system():
    """Test the profile picture system"""
    print("🔧 Testing profile picture system...")

    # Initialize
    db = UserDatabase()
    pm = ProfilePictureManager()

    # Get user
    user = db.get_user_by_username("NumbaNines")
    if user:
        print(f"👤 User: {user.display_name}")
        print(f"🖼️ Current profile: {user.profile_picture}")

        # Check if has profile_picture_type attribute
        if hasattr(user, "profile_picture_type"):
            print(f"📋 Type: {user.profile_picture_type}")
        else:
            print("⚠️ Missing profile_picture_type field")
            # Update to add the field
            success = db.update_user_profile_picture(user.user_id, user.profile_picture, "emoji")
            print(f"✅ Updated: {success}")

        # Test emoji detection
        is_emoji = is_emoji_profile(user.profile_picture)
        print(f"🤔 Is emoji: {is_emoji}")

        # Test default emoji options
        defaults = pm.get_default_emoji_profiles()
        print(f"⚙️ Default options: {len(defaults)} available")

    else:
        print("❌ User not found")


if __name__ == "__main__":
    test_profile_system()
