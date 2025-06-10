# Color update script for SpygateAI Desktop App
import os

def update_colors():
    file_path = 'spygate_desktop_app_tabbed.py'
    
    # Read the file with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Replace all color occurrences
    content = content.replace('#ff6b35', '#1ce783')  # Orange to green
    content = content.replace('#1a1a1a', '#0b0c0f')  # Dark background
    content = content.replace('#121212', '#0b0c0f')  # Darker background
    content = content.replace('#181818', '#0b0c0f')  # Upload container background

    # Also replace the hover color for buttons
    content = content.replace('#e5602e', '#17d474')  # Hover state for green buttons

    # Write back to file with UTF-8 encoding
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print('Color scheme updated successfully!')
    print('Orange (#ff6b35) -> Green (#1ce783)')
    print('Dark colors (#1a1a1a, #121212, #181818) -> New dark (#0b0c0f)')
    print('Hover orange (#e5602e) -> Hover green (#17d474)')

if __name__ == '__main__':
    update_colors() 