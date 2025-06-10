# SpygateAI Logo Assets

## Logo Specifications

### **Recommended Dimensions:**

- **Primary Size**: 180px × 50px (3.6:1 aspect ratio)
- **Maximum Size**: 200px × 60px
- **High-DPI**: 360px × 100px (for retina displays)

### **File Formats:**

- **Primary**: PNG with transparent background
- **Alternative**: SVG (vector format)
- **Backup**: JPG (if transparency not needed)

### **File Naming:**

- `spygate-logo.png` - Primary logo file
- `spygate-logo@2x.png` - High-DPI version (optional)
- `spygate_logo.png` - Alternative naming

### **Design Guidelines:**

- **Background**: Transparent (works on dark theme #0b0c0f)
- **Colors**: Should complement SpygateAI green (#1ce783) or be neutral
- **Style**: Clean, professional, readable at small sizes
- **Content**: Logo should be recognizable at 50px height

### **Placement:**

The logo appears in the top-left sidebar header area:

- Container: 250px wide × 80px tall
- Margins: 20px left/right, 15px top/bottom
- Available space: ~210px wide × 50px tall

### **File Location:**

Place your logo file in one of these locations:

1. `assets/logo/spygate-logo.png` (recommended)
2. `assets/logo/spygate_logo.png`
3. `assets/spygate-logo.png`
4. `logo.png` (root directory)

The application will automatically detect and load your logo!

### **Testing:**

After adding your logo, restart the application to see it loaded.
Check the console for confirmation message:

```
✅ Loaded logo from: assets/logo/spygate-logo.png
```

If no logo is found, the app will use the text fallback:

```
📝 Using text logo (no image found)
```
