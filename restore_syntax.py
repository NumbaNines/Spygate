# Emergency syntax restore script
import re


def restore_syntax():
    file_path = "spygate_desktop_app_tabbed.py"

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    print("🚨 Emergency syntax restoration...")

    # Fix the most critical syntax errors
    content = content.replace(
        "'color: #767676; font-family: 'Segoe UI, sans-serif; font-size: 12px;)",
        "\"color: #767676; font-family: 'Segoe UI', sans-serif; font-size: 12px;\"",
    )

    content = content.replace(
        "'color: #1ce783; font-family: 'Segoe UI, sans-serif; font-size: 28px; font-weight: bold;)",
        "\"color: #1ce783; font-family: 'Segoe UI', sans-serif; font-size: 28px; font-weight: bold;\"",
    )

    content = content.replace(
        "'color: #767676; font-family: 'Segoe UI, sans-serif; font-size: 10px;)",
        "\"color: #767676; font-family: 'Segoe UI', sans-serif; font-size: 10px;\"",
    )

    # Fix docstring quotes
    content = content.replace(
        "'Analysis Tab - Video processing and clip detection.'",
        '"Analysis Tab - Video processing and clip detection."',
    )

    content = content.replace(
        "'Represents a detected clip with metadata.''",
        '"Represents a detected clip with metadata."',
    )

    # Fix missing quotes and broken strings
    content = re.sub(r"QLabel\('([^']*)\)", r'QLabel("\1")', content)
    content = re.sub(r"'Show the ([^']*)'", r'"Show the \1"', content)

    # Fix multi-line strings
    content = re.sub(
        r"setStyleSheet\('([^']*)$", r'setStyleSheet("""\1', content, flags=re.MULTILINE
    )
    content = re.sub(r"^([^']*)'\\)\s*$", r'\1""")', content, flags=re.MULTILINE)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Basic syntax restored! Manual fixes may still be needed.")


if __name__ == "__main__":
    restore_syntax()
