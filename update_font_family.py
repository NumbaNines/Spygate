# Update font family to use Minork Sans
import re


def update_font_family():
    file_path = "spygate_desktop_app_tabbed.py"

    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    print("🔤 Updating font family to Minork Sans...")

    # Replace all font-family declarations with Minork Sans
    content = re.sub(r"font-family: ['\"][^'\"]*['\"]", "font-family: 'Minork Sans'", content)
    content = re.sub(
        r"font-family: ['\"][^'\"]*['\"],\s*['\"][^'\"]*['\"],\s*sans-serif",
        "font-family: 'Minork Sans', sans-serif",
        content,
    )

    # Also replace any remaining Segoe UI references
    content = content.replace("Segoe UI", "Minork Sans")
    content = content.replace("Arial", "Minork Sans")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("✅ Font family updated to Minork Sans!")
    print('📝 All text elements now use: "Minork Sans", sans-serif')


if __name__ == "__main__":
    update_font_family()
