#!/usr/bin/env python3
"""
Update Database Schema
======================

Adds profile_picture_type column to existing database.
"""

import sqlite3


def update_database():
    """Update database schema to include profile_picture_type"""
    db_path = "spygate_users.db"

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Check if column exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        print("Current columns:", columns)

        if "profile_picture_type" not in columns:
            print("Adding profile_picture_type column...")
            cursor.execute('ALTER TABLE users ADD COLUMN profile_picture_type TEXT DEFAULT "emoji"')

            # Update existing users to have emoji type
            cursor.execute(
                'UPDATE users SET profile_picture_type = "emoji" WHERE profile_picture_type IS NULL'
            )

            conn.commit()
            print("✅ Added profile_picture_type column")
        else:
            print("✅ profile_picture_type column already exists")

        # Show updated schema
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        print("Updated columns:", columns)


if __name__ == "__main__":
    update_database()
