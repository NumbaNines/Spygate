# SpygateAI Desktop Application

Modern desktop application for Madden gameplay analysis with a sleek FaceIt-style interface.

## 🚀 Quick Start

### Running the Application

**Primary Method (Recommended):**

```bash
python spygate_desktop.py
```

**Alternative Method:**

```bash
python spygate_desktop_app_faceit_style.py
```

### First Time Setup

1. **Install Dependencies:**

   ```bash
   pip install PyQt6 opencv-python numpy Pillow
   ```

2. **Run the Application:**

   ```bash
   python spygate_desktop.py
   ```

3. **Login:** The app will automatically create a demo user "NumbaNines" with premium access.

## ✨ Features

### 🎮 Modern FaceIt-Style Interface

- **Frameless Window** with custom minimize/maximize/close controls
- **Drag-to-Move** window functionality (drag from top bar)
- **Double-Click** top bar to maximize/restore
- **F11** for fullscreen toggle
- **Custom SpygateAI Logo** integration

### 👤 User Profile System

- **Profile Pictures:** Emoji or custom image uploads
- **Premium Subscriptions:** Upgrade/manage subscription plans
- **User Database:** SQLite-based user management

### 📊 Comprehensive Dashboard

- **Quick Actions:** Upload videos, access play builder, view analysis
- **Analytics Cards:** Videos analyzed, hours processed, formation stats
- **Recent Activity:** Timeline of user actions and analysis
- **Performance Metrics:** Red zone efficiency, conversion rates, etc.

### 🏈 Madden Analysis Tools

- **Video Upload Interface:** Drag-and-drop or browse for Madden gameplay
- **Formation Builder:** Create and manage custom formations
- **Game Planning:** Strategic analysis and planning tools
- **Learning Center:** Tutorials and guides (coming soon)

## 🎨 UI Design

### Color Scheme

- **Background:** Dark theme (#0b0c0f)
- **Accent:** SpygateAI Green (#29d28c)
- **Text:** High contrast (#ffffff, #e3e3e3, #767676)
- **Cards:** Subtle dark (#1a1a1a)

### Typography

- **Font:** Minork Sans (with Arial fallback)
- **Sizing:** Responsive text sizes for different UI elements

## 📁 File Structure

```
SpygateAI/
├── spygate_desktop.py                    # 🚀 MAIN APPLICATION LAUNCHER
├── spygate_desktop_app_faceit_style.py  # Core desktop application
├── user_database.py                     # User management system
├── profile_picture_manager.py           # Profile picture handling
├── assets/
│   └── logo/
│       └── spygate-logo.png            # Custom logo (optional)
├── profile_pictures/                    # User uploaded images
└── spygate_users.db                    # User database
```

## 🔧 Technical Details

### Dependencies

- **PyQt6:** Modern Qt6 framework for desktop UI
- **OpenCV:** Video processing and computer vision
- **NumPy:** Numerical operations
- **Pillow:** Image processing for profile pictures
- **SQLite3:** Built-in database (no additional setup required)

### System Requirements

- **Python:** 3.8 or later
- **OS:** Windows, macOS, or Linux
- **RAM:** 4GB+ recommended
- **Storage:** 100MB+ for application and user data

## 📱 Navigation

### Left Sidebar

- **Dashboard** - Main overview and quick actions
- **Analysis** - Video upload and processing interface
- **Gameplan** - Formation builder and strategic tools
- **Learn** - Tutorials and educational content
- **Clips** - Highlight management
- **Stats** - Performance analytics

### Right Sidebar

- **Context-sensitive** content based on current tab
- **Quick Actions** and **Settings** for active features

### Top Bar

- **Custom Window Controls** (minimize, maximize, close)
- **Profile Picture** with dropdown menu
- **Drag Area** for window movement

## 🎯 User Profiles

### Premium Features

- ⭐ **Unlimited Analysis** - Process unlimited video content
- 🎯 **Advanced AI Coaching** - Detailed insights and recommendations
- 📊 **Custom Reports** - Generate personalized analysis reports
- 🏆 **Beta Features** - Early access to new functionality

### Profile Management

- **Custom Images:** Upload and crop profile pictures
- **Emoji Profiles:** Choose from default emoji options
- **Subscription Management:** Upgrade/downgrade plans
- **Settings:** Personalize application preferences

## 🔒 Data & Privacy

- **Local Database:** All user data stored locally in SQLite
- **Profile Pictures:** Stored in local `profile_pictures/` directory
- **No Cloud Sync:** Data remains on your machine
- **Open Source:** Full transparency in data handling

## 🐛 Troubleshooting

### Common Issues

**"Missing dependencies" error:**

```bash
pip install PyQt6 opencv-python numpy Pillow
```

**"Failed to start" error:**

- Ensure Python 3.8+ is installed
- Check all files are present in the directory
- Try running `python --version` to verify Python installation

**Window controls not working:**

- This is expected behavior for frameless windows
- Use the custom controls in the top-right corner
- Try F11 for fullscreen toggle

### Getting Help

1. Check the error messages for specific guidance
2. Ensure all dependencies are installed
3. Verify file permissions and directory structure
4. Check Python version compatibility

## 🚀 Development

### Architecture

- **Main Window:** `SpygateDesktopFaceItStyle` class
- **User System:** Separate database and profile management
- **Modular Design:** Easy to extend with new features
- **PyQt6 Framework:** Modern, cross-platform UI toolkit

### Key Components

- **Custom Window Controls:** Frameless window implementation
- **Profile System:** User authentication and premium features
- **Dashboard:** Analytics and quick action interface
- **Navigation:** Tab-based interface with context switching

---

**Built with ❤️ for the Madden community**
