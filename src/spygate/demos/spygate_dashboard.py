#!/usr/bin/env python3

"""
SpygateAI Dashboard - FACEIT-Style Layout
========================================
"""

import sys
from pathlib import Path

# Set up proper Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
spygate_path = project_root / "spygate"
sys.path.insert(0, str(spygate_path))

try:
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
    from PyQt6.QtWidgets import *

    print("🏈 SpygateAI Dashboard - Loading...")
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    sys.exit(1)


class SidebarWidget(QWidget):
    """Left sidebar navigation - FACEIT style."""

    tab_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setFixedWidth(280)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border: none;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header with logo - FACEIT style
        header = QWidget()
        header.setFixedHeight(80)
        header.setStyleSheet(
            """
            QWidget {
                background-color: #0f0f0f;
                border-bottom: 1px solid #333;
            }
        """
        )

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(20, 0, 20, 0)

        logo = QLabel("🏈")
        logo.setFont(QFont("Arial", 24))
        logo.setStyleSheet("color: #ff6b35; background: transparent;")
        header_layout.addWidget(logo)

        title = QLabel("SpygateAI")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: white; background: transparent;")
        header_layout.addWidget(title)

        header_layout.addStretch()
        header.setLayout(header_layout)
        layout.addWidget(header)

        # Search bar
        search_container = QWidget()
        search_container.setFixedHeight(60)
        search_container.setStyleSheet("background-color: #1a1a1a; border: none;")

        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(20, 15, 20, 15)

        search_icon = QLabel("🔍")
        search_icon.setStyleSheet("color: #666; background: transparent;")
        search_layout.addWidget(search_icon)

        search_input = QLineEdit()
        search_input.setPlaceholderText("Search")
        search_input.setStyleSheet(
            """
            QLineEdit {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px 12px;
                color: white;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #ff6b35;
            }
        """
        )
        search_layout.addWidget(search_input)

        search_container.setLayout(search_layout)
        layout.addWidget(search_container)

        # Navigation items
        nav_scroll = QScrollArea()
        nav_scroll.setWidgetResizable(True)
        nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        nav_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        nav_scroll.setStyleSheet(
            """
            QScrollArea {
                background-color: #1a1a1a;
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2a2a2a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background-color: #444;
                border-radius: 4px;
            }
        """
        )

        nav_widget = QWidget()
        nav_layout = QVBoxLayout()
        nav_layout.setContentsMargins(0, 10, 0, 10)
        nav_layout.setSpacing(2)

        # Navigation items - FACEIT style
        nav_items = [
            ("▶️", "Auto-Detect", "auto-detect", True),
            ("📚", "Library", "library", False),
            ("📊", "Analysis", "analysis", False),
            ("🔍", "Search", "search", False),
            ("📤", "Export", "export", False),
            ("⚙️", "Settings", "settings", False),
        ]

        self.nav_buttons = []
        for icon, text, action, is_active in nav_items:
            btn = self.create_nav_button(icon, text, action, is_active)
            nav_layout.addWidget(btn)
            self.nav_buttons.append(btn)

        nav_layout.addStretch()
        nav_widget.setLayout(nav_layout)
        nav_scroll.setWidget(nav_widget)

        layout.addWidget(nav_scroll)

        # Bottom section - like FACEIT's upgrade button
        bottom_section = QWidget()
        bottom_section.setFixedHeight(100)
        bottom_section.setStyleSheet(
            """
            QWidget {
                background-color: #0f0f0f;
                border-top: 1px solid #333;
            }
        """
        )

        bottom_layout = QVBoxLayout()
        bottom_layout.setContentsMargins(20, 15, 20, 15)

        upgrade_btn = QPushButton("⚡ Upgrade to Pro")
        upgrade_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ff6b35;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #e55a2b;
            }
        """
        )
        bottom_layout.addWidget(upgrade_btn)

        bottom_section.setLayout(bottom_layout)
        layout.addWidget(bottom_section)

        self.setLayout(layout)

    def create_nav_button(self, icon, text, action, is_active=False):
        """Create a navigation button in FACEIT style."""
        btn = QWidget()
        btn.setFixedHeight(50)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)

        if is_active:
            btn.setStyleSheet(
                """
                QWidget {
                    background-color: #2a2a2a;
                    border-left: 3px solid #ff6b35;
                }
                QWidget:hover {
                    background-color: #333;
                }
            """
            )
        else:
            btn.setStyleSheet(
                """
                QWidget {
                    background-color: transparent;
                    border-left: 3px solid transparent;
                }
                QWidget:hover {
                    background-color: #2a2a2a;
                }
            """
            )

        layout = QHBoxLayout()
        layout.setContentsMargins(20, 0, 20, 0)

        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 16))
        icon_label.setStyleSheet("color: #ccc; background: transparent; border: none;")
        icon_label.setFixedWidth(30)
        layout.addWidget(icon_label)

        text_label = QLabel(text)
        text_label.setFont(
            QFont("Arial", 14, QFont.Weight.Bold if is_active else QFont.Weight.Normal)
        )
        text_label.setStyleSheet(
            f"color: {'white' if is_active else '#ccc'}; background: transparent; border: none;"
        )
        layout.addWidget(text_label)

        btn.setLayout(layout)

        # Add click handler
        btn.mousePressEvent = lambda e, a=action: self.tab_requested.emit(a)

        return btn


class TopHeaderWidget(QWidget):
    """Top header with user info - FACEIT style."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setFixedHeight(80)
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0f0f0f;
                border-bottom: 1px solid #333;
            }
        """
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(30, 0, 30, 0)

        # Premium banner area (like FACEIT's premium banner)
        banner = QWidget()
        banner.setStyleSheet(
            """
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff6b35, stop:1 #e55a2b);
                border-radius: 8px;
            }
        """
        )
        banner.setFixedHeight(50)
        banner.setMinimumWidth(400)

        banner_layout = QHBoxLayout()
        banner_layout.setContentsMargins(20, 0, 20, 0)

        banner_text = QVBoxLayout()
        banner_title = QLabel("Elevate every analysis with SpygateAI Premium")
        banner_title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        banner_title.setStyleSheet("color: white; background: transparent; border: none;")
        banner_text.addWidget(banner_title)

        banner_subtitle = QLabel("Advanced formations • Real-time detection • Pro insights")
        banner_subtitle.setFont(QFont("Arial", 10))
        banner_subtitle.setStyleSheet(
            "color: rgba(255,255,255,0.9); background: transparent; border: none;"
        )
        banner_text.addWidget(banner_subtitle)

        banner_layout.addLayout(banner_text)
        banner_layout.addStretch()

        upgrade_btn = QPushButton("UPGRADE NOW")
        upgrade_btn.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(255,255,255,0.2);
                color: white;
                border: 1px solid rgba(255,255,255,0.3);
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,0.3);
            }
        """
        )
        banner_layout.addWidget(upgrade_btn)

        banner.setLayout(banner_layout)
        layout.addWidget(banner)

        layout.addStretch()

        # User section (like FACEIT's user area)
        user_section = QWidget()
        user_layout = QHBoxLayout()
        user_layout.setContentsMargins(0, 0, 0, 0)

        # Notifications
        notif_btn = QPushButton("🔔")
        notif_btn.setFixedSize(40, 40)
        notif_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2a2a2a;
                border: 1px solid #444;
                border-radius: 20px;
                color: #ccc;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #333;
            }
        """
        )
        user_layout.addWidget(notif_btn)

        # User avatar/profile
        profile_btn = QPushButton("👤")
        profile_btn.setFixedSize(40, 40)
        profile_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ff6b35;
                border: none;
                border-radius: 20px;
                color: white;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e55a2b;
            }
        """
        )
        user_layout.addWidget(profile_btn)

        user_section.setLayout(user_layout)
        layout.addWidget(user_section)

        self.setLayout(layout)


class MainContentWidget(QWidget):
    """Main content area - FACEIT style."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0f0f0f;
                color: white;
            }
        """
        )

        # This will be replaced by the stacked widget content
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)

        placeholder = QLabel("Main content area - will be replaced by stacked widgets")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color: #666; font-size: 16px;")
        layout.addWidget(placeholder)

        self.setLayout(layout)


class AutoDetectContentWidget(QWidget):
    """Auto-detect content - FACEIT style layout."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)

        # Profile section (like FACEIT's user profile area)
        profile_section = QWidget()
        profile_section.setFixedHeight(200)
        profile_section.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
            }
        """
        )

        profile_layout = QHBoxLayout()
        profile_layout.setContentsMargins(30, 20, 30, 20)

        # Left side - Avatar and info
        left_info = QVBoxLayout()

        # Avatar
        avatar = QLabel("🎯")
        avatar.setFont(QFont("Arial", 48))
        avatar.setStyleSheet("color: #ff6b35; background: transparent; border: none;")
        avatar.setFixedSize(80, 80)
        avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_info.addWidget(avatar)

        # Status
        status = QLabel("ANALYSIS READY")
        status.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        status.setStyleSheet("color: #4ade80; background: transparent; border: none;")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_info.addWidget(status)

        profile_layout.addLayout(left_info)

        # Center - Main content
        center_content = QVBoxLayout()

        title = QLabel("Automatic Clip Detection")
        title.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title.setStyleSheet("color: white; background: transparent; border: none;")
        center_content.addWidget(title)

        subtitle = QLabel("Drop a video and let AI automatically detect key football moments")
        subtitle.setFont(QFont("Arial", 14))
        subtitle.setStyleSheet("color: #ccc; background: transparent; border: none;")
        center_content.addWidget(subtitle)

        # Stats row
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(40)

        stats_data = [
            ("📹", "Videos", "0"),
            ("✂️", "Clips", "0"),
            ("🎯", "Accuracy", "95%"),
        ]

        for icon, label, value in stats_data:
            stat_widget = QWidget()
            stat_layout = QVBoxLayout()
            stat_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stat_layout.setSpacing(5)

            stat_icon = QLabel(icon)
            stat_icon.setFont(QFont("Arial", 16))
            stat_icon.setStyleSheet("color: #ff6b35; background: transparent; border: none;")
            stat_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stat_layout.addWidget(stat_icon)

            stat_value = QLabel(value)
            stat_value.setFont(QFont("Arial", 18, QFont.Weight.Bold))
            stat_value.setStyleSheet("color: white; background: transparent; border: none;")
            stat_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stat_layout.addWidget(stat_value)

            stat_label = QLabel(label)
            stat_label.setFont(QFont("Arial", 10))
            stat_label.setStyleSheet("color: #666; background: transparent; border: none;")
            stat_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stat_layout.addWidget(stat_label)

            stat_widget.setLayout(stat_layout)
            stats_layout.addWidget(stat_widget)

        center_content.addLayout(stats_layout)
        profile_layout.addLayout(center_content)

        # Right side - Action button
        action_btn = QPushButton("START ANALYSIS")
        action_btn.setFixedSize(140, 50)
        action_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #ff6b35;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #e55a2b;
            }
        """
        )
        profile_layout.addWidget(action_btn)

        profile_section.setLayout(profile_layout)
        layout.addWidget(profile_section)

        # Bottom content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Drop zone
        left_panel = QWidget()
        left_panel.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
            }
        """
        )

        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(20, 20, 20, 20)

        drop_title = QLabel("Video Drop Zone")
        drop_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        drop_title.setStyleSheet("color: white; background: transparent; border: none;")
        left_layout.addWidget(drop_title)

        drop_area = QLabel("🎮\n\nDrag & drop video files here\n\nSupports MP4, MOV, AVI")
        drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_area.setStyleSheet(
            """
            QLabel {
                background-color: #0f0f0f;
                border: 2px dashed #444;
                border-radius: 8px;
                color: #666;
                font-size: 14px;
                padding: 40px;
                min-height: 200px;
            }
        """
        )
        left_layout.addWidget(drop_area)

        left_panel.setLayout(left_layout)

        # Right panel - Recent activity
        right_panel = QWidget()
        right_panel.setStyleSheet(
            """
            QWidget {
                background-color: #1a1a1a;
                border: 1px solid #333;
                border-radius: 12px;
            }
        """
        )

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(20, 20, 20, 20)

        activity_title = QLabel("Recent Analysis")
        activity_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        activity_title.setStyleSheet("color: white; background: transparent; border: none;")
        right_layout.addWidget(activity_title)

        activity_list = QLabel("No recent analysis\n\nStart by uploading a video")
        activity_list.setAlignment(Qt.AlignmentFlag.AlignCenter)
        activity_list.setStyleSheet(
            """
            QLabel {
                background-color: #0f0f0f;
                border: 1px solid #333;
                border-radius: 8px;
                color: #666;
                font-size: 14px;
                padding: 40px;
                min-height: 200px;
            }
        """
        )
        right_layout.addWidget(activity_list)

        right_panel.setLayout(right_layout)

        content_splitter.addWidget(left_panel)
        content_splitter.addWidget(right_panel)
        content_splitter.setSizes([1, 1])

        layout.addWidget(content_splitter)
        self.setLayout(layout)


class SpygateMainWindow(QMainWindow):
    """Main SpygateAI window - FACEIT style."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI")
        self.setGeometry(100, 100, 1600, 1000)
        self.init_ui()

    def init_ui(self):
        # Set dark theme
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #0f0f0f;
                color: white;
            }
        """
        )

        central = QWidget()
        self.setCentralWidget(central)

        # Main layout - horizontal split like FACEIT
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left sidebar
        self.sidebar = SidebarWidget()
        self.sidebar.tab_requested.connect(self.switch_tab)
        main_layout.addWidget(self.sidebar)

        # Right content area
        right_area = QWidget()
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Top header
        self.header = TopHeaderWidget()
        right_layout.addWidget(self.header)

        # Main content stack
        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet("background-color: #0f0f0f;")

        # Add content widgets
        self.content_stack.addWidget(AutoDetectContentWidget())  # 0
        self.content_stack.addWidget(
            self.create_placeholder("📚", "Library", "Import and manage your video library")
        )  # 1
        self.content_stack.addWidget(
            self.create_placeholder("📊", "Analysis", "Formation and play analysis tools")
        )  # 2
        self.content_stack.addWidget(
            self.create_placeholder("🔍", "Search", "Advanced search and filtering")
        )  # 3
        self.content_stack.addWidget(
            self.create_placeholder("📤", "Export", "Export clips and analysis data")
        )  # 4
        self.content_stack.addWidget(
            self.create_placeholder("⚙️", "Settings", "Configuration and preferences")
        )  # 5

        right_layout.addWidget(self.content_stack)
        right_area.setLayout(right_layout)

        main_layout.addWidget(right_area)
        central.setLayout(main_layout)

    def create_placeholder(self, icon, title, description):
        """Create placeholder content for other tabs."""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 48))
        icon_label.setStyleSheet("color: #ff6b35; background: transparent; border: none;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setStyleSheet("color: white; background: transparent; border: none;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setFont(QFont("Arial", 14))
        desc_label.setStyleSheet("color: #666; background: transparent; border: none;")
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)

        widget.setLayout(layout)
        return widget

    def switch_tab(self, tab_name):
        """Switch between different sections."""
        tab_mapping = {
            "auto-detect": 0,
            "library": 1,
            "analysis": 2,
            "search": 3,
            "export": 4,
            "settings": 5,
        }

        if tab_name in tab_mapping:
            self.content_stack.setCurrentIndex(tab_mapping[tab_name])


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("SpygateAI")

    # Set application style to Fusion for better dark theme support
    app.setStyle("Fusion")

    window = SpygateMainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
