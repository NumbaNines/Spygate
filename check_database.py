#!/usr/bin/env python3
"""
Check SpygateAI Database Contents
================================

Display all users and subscriptions in the database.
"""

import json

from user_database import UserDatabase


def main():
    db = UserDatabase()

    print("🏈 SpygateAI User Database Contents")
    print("=" * 50)

    # Get all users (we need to query the database directly for this)
    import sqlite3

    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()

        # Users table
        print("\n👥 USERS:")
        cursor.execute("SELECT * FROM users")
        users = cursor.fetchall()

        for user in users:
            print(f"  ID: {user[0]}")
            print(f"  Username: {user[1]}")
            print(f"  Display Name: {user[2]}")
            print(f"  Email: {user[3]}")
            print(f"  Profile Picture: {user[5]}")
            print(f"  Created: {user[6][:10]}")
            print(f"  Last Login: {user[7][:10]}")
            print(f"  Is Premium: {'Yes' if user[8] else 'No'}")
            print(f"  Subscription Type: {user[9]}")
            print(f"  Subscription Expires: {user[10] or 'Never'}")
            print("-" * 30)

        # Subscriptions table
        print("\n💳 SUBSCRIPTIONS:")
        cursor.execute("SELECT * FROM subscriptions")
        subscriptions = cursor.fetchall()

        for sub in subscriptions:
            print(f"  Subscription ID: {sub[0]}")
            print(f"  User ID: {sub[1]}")
            print(f"  Plan Type: {sub[2]}")
            print(f"  Status: {sub[3]}")
            print(f"  Started: {sub[4][:10]}")
            print(f"  Expires: {sub[5][:10] if sub[5] else 'Never'}")
            print(f"  Price Paid: ${sub[6]:.2f}")

            # Parse features JSON
            try:
                features = json.loads(sub[7])
                print(f"  Features: {len(features)} features available")
                for key, value in features.items():
                    if isinstance(value, bool):
                        print(f"    {key}: {'✅' if value else '❌'}")
                    else:
                        print(f"    {key}: {value}")
            except:
                print(f"  Features: {sub[7]}")

            print("-" * 30)

        # User settings
        print("\n⚙️ USER SETTINGS:")
        cursor.execute("SELECT * FROM user_settings")
        settings = cursor.fetchall()

        for setting in settings:
            print(f"  User ID {setting[1]}: {setting[2]} = {setting[3]}")

        if not settings:
            print("  No user settings found.")


if __name__ == "__main__":
    main()
