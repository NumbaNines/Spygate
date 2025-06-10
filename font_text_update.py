# Font and text color update script for SpygateAI Desktop App
import os
import re

def update_text_styling():
    file_path = 'spygate_desktop_app_tabbed.py'
    
    # Read the file with UTF-8 encoding
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print("🎨 Updating text colors and fonts...")
    
    # 1. Update button text colors to #e3e3e3
    # Match QPushButton color declarations
    button_color_pattern = r'(QPushButton\s*\{[^}]*color:\s*)[^;]+;'
    content = re.sub(button_color_pattern, r'\1#e3e3e3;', content)
    
    # 2. Update header text in tabs to #ffffff  
    # Headers are typically larger font sizes (18px+) or have "header" in the variable name
    header_patterns = [
        r'(QLabel\s*\{[^}]*font-size:\s*(?:18|20|22|24|26|28|30)px[^}]*color:\s*)[^;]+;',
        r'(header[^=]*setStyleSheet[^}]*color:\s*)[^;]+;'
    ]
    
    for pattern in header_patterns:
        content = re.sub(pattern, r'\1#ffffff;', content)
    
    # 3. Update field guide/placeholder text to #767676
    # These are typically smaller fonts (12px, 14px) or have guide/hint in name
    guide_patterns = [
        r'(QLabel\s*\{[^}]*font-size:\s*(?:12|14)px[^}]*color:\s*)[^;]+;',
        r'(placeholder[^=]*setStyleSheet[^}]*color:\s*)[^;]+;',
        r'(guide[^=]*setStyleSheet[^}]*color:\s*)[^;]+;',
        r'(hint[^=]*setStyleSheet[^}]*color:\s*)[^;]+;'
    ]
    
    for pattern in guide_patterns:
        content = re.sub(pattern, r'\1#767676;', content)

    # 4. Add font-family: "Segoe UI" (closest to Minork Sans) to all text elements
    # Update QLabel, QPushButton, and other text elements
    font_patterns = [
        r'(QLabel\s*\{[^}]*)(font-size:[^;]+;)',
        r'(QPushButton\s*\{[^}]*)(font-size:[^;]+;)',
        r'(QWidget\s*\{[^}]*)(font-size:[^;]+;)'
    ]
    
    for pattern in font_patterns:
        content = re.sub(pattern, r'\1font-family: "Segoe UI", "Arial", sans-serif; \2', content)

    # Specific updates for known text elements that need different colors
    specific_updates = [
        # Stats line and metadata should be guide text
        (r'(stats_line[^=]*setStyleSheet[^}]*color:\s*)[^;]+;', r'\1#767676;'),
        (r'(info_label[^=]*setStyleSheet[^}]*color:\s*)[^;]+;', r'\1#767676;'),
        
        # Tab headers should be white
        (r'(QLabel[^}]*font-size:\s*24px[^}]*color:\s*)[^;]+;', r'\1#ffffff;'),
        (r'(QLabel[^}]*font-size:\s*20px[^}]*color:\s*)[^;]+;', r'\1#ffffff;'),
        
        # Button text should be #e3e3e3
        (r'(QPushButton[^}]*color:\s*)white;', r'\1#e3e3e3;'),
        (r'(QPushButton[^}]*color:\s*)[^;]+;', r'\1#e3e3e3;'),
    ]
    
    for old_pattern, new_pattern in specific_updates:
        content = re.sub(old_pattern, new_pattern, content)

    # Write back to file with UTF-8 encoding
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print('✅ Text styling updated successfully!')
    print('📝 Changes applied:')
    print('   • Button text: #e3e3e3')
    print('   • Header text: #ffffff') 
    print('   • Guide text: #767676')
    print('   • Font family: "Segoe UI" (modern sans-serif)')

if __name__ == "__main__":
    update_text_styling() 