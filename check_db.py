#!/usr/bin/env python3

import sqlite3


def check_database():
    conn = sqlite3.connect("madden_ocr_training.db")
    cursor = conn.cursor()

    # Get table schema
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    print("Database Schema:")
    for table in tables:
        print(f"  {table[0]}")

    # Get table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    table_names = [row[0] for row in cursor.fetchall()]
    print(f"\nTables: {table_names}")

    # Check each table
    for table_name in table_names:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"\n{table_name}: {count} rows")

        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            rows = cursor.fetchall()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in cursor.fetchall()]
            print(f"  Columns: {columns}")
            print("  Sample rows:")
            for row in rows:
                print(f"    {row}")

    conn.close()


if __name__ == "__main__":
    check_database()
