import os
import sys

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class SpygateDesktopFaceIt(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SpygateAI Desktop - FaceIt Style")
        self.setGeometry(100, 100, 1400, 900)

        # Set dark background
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: #0b0c0f;
                color: #e3e3e3;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )

        self.init_ui()

    def init_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout (3-column)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left Sidebar
        self.create_left_sidebar(main_layout)

        # Main Content Area
        self.create_main_content(main_layout)

        # Right Sidebar
        self.create_right_sidebar(main_layout)

    def create_left_sidebar(self, parent_layout):
        # Left sidebar frame
        left_sidebar = QFrame()
        left_sidebar.setFixedWidth(250)
        left_sidebar.setStyleSheet(
            f"""
            QFrame {{
                background-color: #0b0c0f;
                border-right: 1px solid #2a2a2a;
            }}
        """
        )

        sidebar_layout = QVBoxLayout(left_sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Logo/Header area
        header_widget = QWidget()
        header_widget.setFixedHeight(80)
        header_widget.setStyleSheet(
            f"""
            QWidget {{
                background-color: #0b0c0f;
                border-bottom: 1px solid #2a2a2a;
            }}
        """
        )
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 0, 20, 0)

        logo_label = QLabel("SPYGATE")
        logo_label.setStyleSheet(
            f"""
            QLabel {{
                color: #1ce783;
                font-size: 24px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )
        header_layout.addWidget(logo_label)
        header_layout.addStretch()

        sidebar_layout.addWidget(header_widget)

        # Navigation items
        nav_items = [
            ("🔍", "Search"),
            ("👥", "Party Finder"),
            ("▶️", "Play"),
            ("🏆", "Rank"),
            ("📊", "Track"),
            ("👁️", "Watch"),
            ("📰", "Feed"),
            ("🏛️", "Clubs"),
            ("➕", "Create a Club"),
            ("🎯", "Missions"),
            ("🛒", "Shop"),
            ("⬆️", "Upgrade"),
        ]

        nav_widget = QWidget()
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 20, 0, 0)
        nav_layout.setSpacing(5)

        # Store nav buttons for selection management
        self.nav_buttons = []
        
        for icon, text in nav_items:
            nav_button = self.create_nav_button(icon, text)
            nav_layout.addWidget(nav_button)
            self.nav_buttons.append(nav_button)

        nav_layout.addStretch()
        sidebar_layout.addWidget(nav_widget)

        parent_layout.addWidget(left_sidebar)

    def create_nav_button(self, icon, text):
        button = QPushButton(f"{icon}  {text}")
        button.setFixedHeight(45)
        button.setCheckable(True)  # Make button checkable for selected state
        
        # Set first button (Search) as selected by default
        if text == "Search":
            button.setChecked(True)
            
        button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: #767676;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                text-align: left;
                padding-left: 20px;
                border: none;
                border-radius: 0px;
            }}
            QPushButton:checked {{
                color: #ffffff;
                background-color: #1a1a1a;
            }}
            QPushButton:hover {{
                background-color: #1a1a1a;
                color: #1ce783;
            }}
            QPushButton:pressed {{
                background-color: #1ce783;
                color: #0b0c0f;
            }}
        """
        )
        
        # Connect button click to handle selection
        button.clicked.connect(lambda: self.handle_nav_selection(button))
        return button

    def handle_nav_selection(self, selected_button):
        """Handle navigation tab selection - only one tab selected at a time"""
        for button in self.nav_buttons:
            button.setChecked(False)
        selected_button.setChecked(True)

    def create_main_content(self, parent_layout):
        # Main content area
        main_content = QFrame()
        main_content.setStyleSheet(
            f"""
            QFrame {{
                background-color: #0b0c0f;
            }}
        """
        )

        content_layout = QVBoxLayout(main_content)
        content_layout.setContentsMargins(30, 30, 30, 30)
        content_layout.setSpacing(20)

        # Premium banner (like FaceIt Premium)
        self.create_premium_banner(content_layout)

        # Main game/analysis area
        self.create_game_area(content_layout)

        # Bottom content sections
        self.create_bottom_sections(content_layout)

        parent_layout.addWidget(main_content, 1)  # Takes remaining space

    def create_premium_banner(self, parent_layout):
        banner = QFrame()
        banner.setFixedHeight(120)
        banner.setStyleSheet(
            f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1ce783, stop:1 #0b4f2a);
                border-radius: 15px;
                margin-bottom: 10px;
            }}
        """
        )

        banner_layout = QHBoxLayout(banner)
        banner_layout.setContentsMargins(30, 20, 30, 20)

        # Text content
        text_widget = QWidget()
        text_layout = QVBoxLayout(text_widget)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(5)

        title_label = QLabel("Elevate every match with SpygateAI Premium")
        title_label.setStyleSheet(
            f"""
            QLabel {{
                color: #ffffff;
                font-size: 28px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )

        subtitle_label = QLabel("Analyze better • Strategize faster • Win more")
        subtitle_label.setStyleSheet(
            f"""
            QLabel {{
                color: #ffffff;
                font-size: 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin-top: 5px;
            }}
        """
        )

        upgrade_button = QPushButton("UPGRADE NOW")
        upgrade_button.setFixedSize(150, 40)
        upgrade_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #ffffff;
                color: #0b0c0f;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                border: none;
                border-radius: 8px;
                margin-top: 10px;
            }}
            QPushButton:hover {{
                background-color: #f0f0f0;
            }}
        """
        )

        text_layout.addWidget(title_label)
        text_layout.addWidget(subtitle_label)
        text_layout.addWidget(upgrade_button, alignment=Qt.AlignmentFlag.AlignLeft)

        banner_layout.addWidget(text_widget)
        banner_layout.addStretch()

        # Character images placeholder
        char_label = QLabel("🏈⚡🎯")
        char_label.setStyleSheet(
            f"""
            QLabel {{
                color: #ffffff;
                font-size: 48px;
            }}
        """
        )
        banner_layout.addWidget(char_label)

        parent_layout.addWidget(banner)

    def create_game_area(self, parent_layout):
        # Game area (like the match interface in FaceIt)
        game_frame = QFrame()
        game_frame.setFixedHeight(300)
        game_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: #1a1a1a;
                border-radius: 15px;
                border: 1px solid #2a2a2a;
            }}
        """
        )

        game_layout = QVBoxLayout(game_frame)
        game_layout.setContentsMargins(30, 30, 30, 30)

        # Game header
        game_header = QHBoxLayout()

        game_icon = QLabel("🎮")
        game_icon.setStyleSheet(
            f"""
            QLabel {{
                font-size: 32px;
                color: #1ce783;
            }}
        """
        )

        game_info = QWidget()
        info_layout = QVBoxLayout(game_info)
        info_layout.setContentsMargins(15, 0, 0, 0)
        info_layout.setSpacing(5)

        game_title = QLabel("MADDEN 25")
        game_title.setStyleSheet(
            f"""
            QLabel {{
                color: #ffffff;
                font-size: 18px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )

        skill_level = QLabel("SKILL LEVEL: ⭐ 620")
        skill_level.setStyleSheet(
            f"""
            QLabel {{
                color: #1ce783;
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )

        info_layout.addWidget(game_title)
        info_layout.addWidget(skill_level)

        game_header.addWidget(game_icon)
        game_header.addWidget(game_info)
        game_header.addStretch()

        game_layout.addLayout(game_header)

        # Analysis status
        status_label = QLabel("Ready for Analysis")
        status_label.setStyleSheet(
            f"""
            QLabel {{
                color: #767676;
                font-size: 16px;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin: 20px 0;
            }}
        """
        )
        game_layout.addWidget(status_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Action buttons
        button_layout = QHBoxLayout()

        upload_button = QPushButton("📁 Upload Video")
        upload_button.setFixedSize(150, 40)
        upload_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #1ce783;
                color: #0b0c0f;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: #16b870;
            }}
        """
        )

        analyze_button = QPushButton("🔍 Quick Analyze")
        analyze_button.setFixedSize(150, 40)
        analyze_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: #1ce783;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                border: 2px solid #1ce783;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: #1ce783;
                color: #0b0c0f;
            }}
        """
        )

        button_layout.addStretch()
        button_layout.addWidget(upload_button)
        button_layout.addWidget(analyze_button)
        button_layout.addStretch()

        game_layout.addLayout(button_layout)
        game_layout.addStretch()

        parent_layout.addWidget(game_frame)

    def create_bottom_sections(self, parent_layout):
        # Bottom sections (Latest Post and Watch)
        bottom_layout = QHBoxLayout()

        # Latest Post section
        post_frame = QFrame()
        post_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: #1a1a1a;
                border-radius: 10px;
                border: 1px solid #2a2a2a;
            }}
        """
        )

        post_layout = QVBoxLayout(post_frame)
        post_layout.setContentsMargins(20, 20, 20, 20)

        post_title = QLabel("Latest Analysis")
        post_title.setStyleSheet(
            f"""
            QLabel {{
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin-bottom: 10px;
            }}
        """
        )

        post_content = QLabel("No recent analysis available")
        post_content.setStyleSheet(
            f"""
            QLabel {{
                color: #767676;
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )

        post_layout.addWidget(post_title)
        post_layout.addWidget(post_content)
        post_layout.addStretch()

        # Watch section
        watch_frame = QFrame()
        watch_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: #1a1a1a;
                border-radius: 10px;
                border: 1px solid #2a2a2a;
            }}
        """
        )

        watch_layout = QVBoxLayout(watch_frame)
        watch_layout.setContentsMargins(20, 20, 20, 20)

        watch_title = QLabel("Watch")
        watch_title.setStyleSheet(
            f"""
            QLabel {{
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin-bottom: 10px;
            }}
        """
        )

        watch_content = QLabel("Pro Player Analysis Coming Soon")
        watch_content.setStyleSheet(
            f"""
            QLabel {{
                color: #767676;
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )

        watch_layout.addWidget(watch_title)
        watch_layout.addWidget(watch_content)
        watch_layout.addStretch()

        bottom_layout.addWidget(post_frame)
        bottom_layout.addWidget(watch_frame)

        parent_layout.addLayout(bottom_layout)

    def create_right_sidebar(self, parent_layout):
        # Right sidebar
        right_sidebar = QFrame()
        right_sidebar.setFixedWidth(300)
        right_sidebar.setStyleSheet(
            f"""
            QFrame {{
                background-color: #0b0c0f;
                border-left: 1px solid #2a2a2a;
            }}
        """
        )

        sidebar_layout = QVBoxLayout(right_sidebar)
        sidebar_layout.setContentsMargins(20, 30, 20, 30)
        sidebar_layout.setSpacing(20)

        # Parties section
        self.create_parties_section(sidebar_layout)

        # Clubs section
        self.create_clubs_section(sidebar_layout)

        # Discover section
        self.create_discover_section(sidebar_layout)

        sidebar_layout.addStretch()

        parent_layout.addWidget(right_sidebar)

    def create_parties_section(self, parent_layout):
        parties_frame = QFrame()
        parties_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: #1a1a1a;
                border-radius: 10px;
                border: 1px solid #2a2a2a;
            }}
        """
        )

        parties_layout = QVBoxLayout(parties_frame)
        parties_layout.setContentsMargins(15, 15, 15, 15)

        header_layout = QHBoxLayout()

        parties_title = QLabel("Parties")
        parties_title.setStyleSheet(
            f"""
            QLabel {{
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )

        party_count = QLabel("2")
        party_count.setFixedSize(25, 25)
        party_count.setStyleSheet(
            f"""
            QLabel {{
                background-color: #1ce783;
                color: #0b0c0f;
                font-size: 12px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                border-radius: 12px;
                text-align: center;
            }}
        """
        )
        party_count.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header_layout.addWidget(parties_title)
        header_layout.addStretch()
        header_layout.addWidget(party_count)

        parties_layout.addLayout(header_layout)

        # Party items
        party_item = QFrame()
        party_item.setFixedHeight(50)
        party_item.setStyleSheet(
            f"""
            QFrame {{
                background-color: #2a2a2a;
                border-radius: 8px;
                margin: 5px 0;
            }}
        """
        )

        party_item_layout = QHBoxLayout(party_item)
        party_item_layout.setContentsMargins(10, 10, 10, 10)

        party_avatars = QLabel("👥")
        party_avatars.setStyleSheet(
            f"""
            QLabel {{
                font-size: 20px;
            }}
        """
        )

        party_info = QLabel("Analysis Team")
        party_info.setStyleSheet(
            f"""
            QLabel {{
                color: #e3e3e3;
                font-size: 14px;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )

        party_count_small = QLabel("6")
        party_count_small.setStyleSheet(
            f"""
            QLabel {{
                color: #767676;
                font-size: 12px;
                font-family: 'Minork Sans', Arial, sans-serif;
            }}
        """
        )

        party_item_layout.addWidget(party_avatars)
        party_item_layout.addWidget(party_info)
        party_item_layout.addStretch()
        party_item_layout.addWidget(party_count_small)

        parties_layout.addWidget(party_item)

        # Party Finder button
        finder_button = QPushButton("PARTY FINDER")
        finder_button.setFixedHeight(35)
        finder_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #1ce783;
                color: #0b0c0f;
                font-size: 12px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                border: none;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: #16b870;
            }}
        """
        )

        parties_layout.addWidget(finder_button)

        parent_layout.addWidget(parties_frame)

    def create_clubs_section(self, parent_layout):
        clubs_frame = QFrame()
        clubs_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: #1a1a1a;
                border-radius: 10px;
                border: 1px solid #2a2a2a;
            }}
        """
        )

        clubs_layout = QVBoxLayout(clubs_frame)
        clubs_layout.setContentsMargins(15, 15, 15, 15)

        clubs_title = QLabel("Clubs")
        clubs_title.setStyleSheet(
            f"""
            QLabel {{
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                margin-bottom: 10px;
            }}
        """
        )

        clubs_layout.addWidget(clubs_title)

        # Club items
        club_items = [("MCS Club", "102,929"), ("Pro Analysis", "31,939")]

        for club_name, member_count in club_items:
            club_item = QFrame()
            club_item.setFixedHeight(45)
            club_item.setStyleSheet(
                f"""
                QFrame {{
                    background-color: #2a2a2a;
                    border-radius: 8px;
                    margin: 3px 0;
                }}
                QFrame:hover {{
                    background-color: #333333;
                }}
            """
            )

            club_layout = QHBoxLayout(club_item)
            club_layout.setContentsMargins(10, 10, 10, 10)

            club_icon = QLabel("🏛️")
            club_icon.setStyleSheet("font-size: 16px;")

            club_info_widget = QWidget()
            club_info_layout = QVBoxLayout(club_info_widget)
            club_info_layout.setContentsMargins(8, 0, 0, 0)
            club_info_layout.setSpacing(2)

            club_name_label = QLabel(club_name)
            club_name_label.setStyleSheet(
                f"""
                QLabel {{
                    color: #e3e3e3;
                    font-size: 13px;
                    font-weight: bold;
                    font-family: 'Minork Sans', Arial, sans-serif;
                }}
            """
            )

            club_members_label = QLabel(f"Members: {member_count}")
            club_members_label.setStyleSheet(
                f"""
                QLabel {{
                    color: #767676;
                    font-size: 11px;
                    font-family: 'Minork Sans', Arial, sans-serif;
                }}
            """
            )

            club_info_layout.addWidget(club_name_label)
            club_info_layout.addWidget(club_members_label)

            club_layout.addWidget(club_icon)
            club_layout.addWidget(club_info_widget)
            club_layout.addStretch()

            clubs_layout.addWidget(club_item)

        parent_layout.addWidget(clubs_frame)

    def create_discover_section(self, parent_layout):
        discover_button = QPushButton("DISCOVER CLUBS")
        discover_button.setFixedHeight(40)
        discover_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: #1ce783;
                font-size: 14px;
                font-weight: bold;
                font-family: 'Minork Sans', Arial, sans-serif;
                border: 2px solid #1ce783;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: #1ce783;
                color: #0b0c0f;
            }}
        """
        )

        parent_layout.addWidget(discover_button)


def main():
    app = QApplication(sys.argv)

    # Set application-wide font
    font = QFont("Minork Sans", 10)
    app.setFont(font)

    window = SpygateDesktopFaceIt()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
