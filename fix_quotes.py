# Fix nested quotes in style sheets
import re

def fix_quotes():
    file_path = 'spygate_desktop_app_tabbed.py'
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print("🔧 Fixing quote syntax errors...")
    
    # Fix nested quotes in font-family declarations
    content = re.sub(r'font-family: "([^"]*)"([^"]*)"([^"]*)"', r"font-family: '\1\2\3'", content)
    content = re.sub(r'font-family: "Segoe UI", sans-serif', r"font-family: 'Segoe UI', sans-serif", content)
    content = re.sub(r'font-family: "Arial", sans-serif', r"font-family: 'Arial', sans-serif", content)
    
    # Fix any remaining problematic nested quotes
    content = re.sub(r'"([^"]*)"([^"]*)"([^"]*)"', r"'\1\2\3'", content)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print('✅ Quote syntax fixed!')

if __name__ == "__main__":
    fix_quotes() 