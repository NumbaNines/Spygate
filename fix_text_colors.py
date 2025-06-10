# Targeted text color fix script for SpygateAI Desktop App
import re


def fix_text_colors():
    file_path = "spygate_desktop_app_tabbed.py"

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    print("🎨 Fixing specific text colors...")

    # Fix guide/subtitle text (small text that describes what to input)
    content = content.replace(
        "color: #ccc; font-size: 12px;",
        'color: #767676; font-family: "Segoe UI", sans-serif; font-size: 12px;',
    )
    content = content.replace(
        "color: #888; font-size: 10px;",
        'color: #767676; font-family: "Segoe UI", sans-serif; font-size: 10px;',
    )
    content = content.replace(
        "color: #aaa;", 'color: #767676; font-family: "Segoe UI", sans-serif;'
    )
    content = content.replace(
        "color: #666;", 'color: #767676; font-family: "Segoe UI", sans-serif;'
    )

    # Fix header text (larger headers in tabs)
    content = content.replace(
        "color: #1ce783; font-size: 16px; font-weight: bold;",
        'color: #ffffff; font-family: "Segoe UI", sans-serif; font-size: 16px; font-weight: bold;',
    )
    content = content.replace(
        "color: white;", 'color: #ffffff; font-family: "Segoe UI", sans-serif;'
    )

    # Fix button text (ensure all buttons use #e3e3e3)
    content = re.sub(r"(QPushButton[^}]*color:\s*)[^;#]+([#\w]+)?;", r"\1#e3e3e3;", content)

    # Fix activity background from white to dark
    content = content.replace("background-color: #ffffff;", "background-color: #0b0c0f;")
    content = content.replace("background-color: #ffffff", "background-color: #0b0c0f")

    # Fix upload container background
    content = content.replace(
        "background-color: #ffffff;\n                border-radius: 12px;",
        "background-color: #0b0c0f;\n                border-radius: 12px;",
    )

    # Ensure all text has proper font family
    content = re.sub(r"(font-size: \d+px;)", r'font-family: "Segoe UI", sans-serif; \1', content)

    # Remove duplicate font-family declarations
    content = re.sub(
        r'(font-family: "[^"]+", sans-serif;\s*)+',
        r'font-family: "Segoe UI", sans-serif; ',
        content,
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Text colors fixed successfully!")
    print("📝 Applied corrections:")
    print("   • Guide text: #767676")
    print("   • Headers: #ffffff")
    print("   • Button text: #e3e3e3")
    print("   • Background colors fixed")
    print('   • Font family: "Segoe UI"')


if __name__ == "__main__":
    fix_text_colors()
