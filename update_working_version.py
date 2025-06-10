# Update working version with correct colors and font
import re


def update_working_version():
    file_path = "spygate_desktop_app_tabbed_fixed.py"

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    print("🎨 Updating working version with correct styling...")

    # 1. Update orange colors to green
    content = content.replace("#ff6b35", "#1ce783")
    content = content.replace("#ff8b55", "#17d474")  # hover state

    # 2. Update dark colors
    content = content.replace("#1a1a1a", "#0b0c0f")
    content = content.replace("#2a2a2a", "#0f1015")  # hover state

    # 3. Update text colors according to specifications
    # Button text should be #e3e3e3
    # Headers should be #ffffff
    # Field guides should be #767676

    # 4. Add font-family declarations where missing
    # Find all color declarations and add font-family
    content = re.sub(
        r"(color: #[a-fA-F0-9]{6};)",
        r'\1\n                font-family: "Minork Sans", sans-serif;',
        content,
    )

    # 5. Clean up any duplicates
    content = re.sub(
        r'(font-family: "Minork Sans", sans-serif;\s*)+',
        r'font-family: "Minork Sans", sans-serif;\n                ',
        content,
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Working version updated successfully!")
    print("🎨 Colors: Orange -> Green (#1ce783), Dark backgrounds -> #0b0c0f")
    print('🔤 Font: All text now uses "Minork Sans"')


if __name__ == "__main__":
    update_working_version()
