from PyQt6.QtChart import (
    QBarCategoryAxis,
    QBarSeries,
    QBarSet,
    QChart,
    QChartView,
    QValueAxis,
)
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class AnalyticsView(QWidget):
    """Analytics view component for displaying performance statistics and charts."""

    def __init__(self):
        super().__init__()
        self._setup_ui()

    def _setup_ui(self):
        """Set up the analytics view UI."""
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(24)

        # Header section
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Analytics")
        title.setStyleSheet(
            """
            font-size: 24px;
            font-weight: bold;
            color: #FFFFFF;
        """
        )
        header_layout.addWidget(title)

        # Filter section
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.setSpacing(16)

        player_label = QLabel("Player:")
        player_label.setStyleSheet("color: #D1D5DB;")
        filter_layout.addWidget(player_label)

        self.player_combo = QComboBox()
        self.player_combo.setStyleSheet(
            """
            QComboBox {
                background: #1E1E1E;
                border: 1px solid #3B3B3B;
                border-radius: 4px;
                color: #FFFFFF;
                padding: 6px;
                min-width: 150px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: url(assets/down-arrow.png);
            }
        """
        )
        self.player_combo.addItems(
            ["All Players", "Self", "Opponent: John", "Opponent: Mike"]
        )
        filter_layout.addWidget(self.player_combo)

        filter_layout.addStretch()

        export_button = QPushButton("Export Data")
        export_button.setStyleSheet(
            """
            QPushButton {
                background: #3B82F6;
                color: #FFFFFF;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #2563EB;
            }
        """
        )
        filter_layout.addWidget(export_button)

        header_layout.addWidget(filter_widget)
        layout.addWidget(header)

        # Stats grid
        stats_grid = QFrame()
        stats_grid.setStyleSheet(
            """
            QFrame {
                background: #2A2A2A;
                border-radius: 8px;
            }
        """
        )
        stats_layout = QHBoxLayout(stats_grid)
        stats_layout.setContentsMargins(24, 24, 24, 24)
        stats_layout.setSpacing(24)

        # Add stat cards
        stats = [
            ("Total Clips", "156"),
            ("Avg. Play Duration", "12.5s"),
            ("Success Rate", "68%"),
            ("Common Formation", "Shotgun"),
        ]

        for title, value in stats:
            stat_card = QFrame()
            stat_card.setStyleSheet(
                """
                QFrame {
                    background: #1E1E1E;
                    border-radius: 4px;
                    padding: 16px;
                }
            """
            )
            card_layout = QVBoxLayout(stat_card)

            value_label = QLabel(value)
            value_label.setStyleSheet(
                """
                font-size: 24px;
                font-weight: bold;
                color: #FFFFFF;
            """
            )
            card_layout.addWidget(value_label)

            title_label = QLabel(title)
            title_label.setStyleSheet("color: #D1D5DB;")
            card_layout.addWidget(title_label)

            stats_layout.addWidget(stat_card)

        layout.addWidget(stats_grid)

        # Charts section
        charts_widget = QWidget()
        charts_layout = QHBoxLayout(charts_widget)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.setSpacing(24)

        # Play frequency chart
        play_chart = self._create_play_frequency_chart()
        charts_layout.addWidget(play_chart)

        # Success rate chart
        success_chart = self._create_success_rate_chart()
        charts_layout.addWidget(success_chart)

        layout.addWidget(charts_widget)

        # Add stretch to push content to top
        layout.addStretch()

    def _create_play_frequency_chart(self):
        """Create a chart showing play frequency."""
        chart = QChart()
        chart.setTitle("Play Frequency")
        chart.setTitleBrush(Qt.GlobalColor.white)
        chart.setBackgroundBrush(Qt.GlobalColor.transparent)

        # Create series
        series = QBarSeries()
        plays = QBarSet("Plays")
        plays.append([30, 25, 20, 15, 10])
        series.append(plays)

        # Add series to chart
        chart.addSeries(series)

        # Set up axes
        categories = ["Run Inside", "Pass Short", "Pass Deep", "Screen", "Option"]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, 35)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)

        # Create chart view
        chart_view = QChartView(chart)
        chart_view.setRenderHint(chart_view.RenderHint.Antialiasing)
        chart_view.setStyleSheet(
            """
            background: #2A2A2A;
            border-radius: 8px;
        """
        )

        return chart_view

    def _create_success_rate_chart(self):
        """Create a chart showing success rates."""
        chart = QChart()
        chart.setTitle("Success Rate by Formation")
        chart.setTitleBrush(Qt.GlobalColor.white)
        chart.setBackgroundBrush(Qt.GlobalColor.transparent)

        # Create series
        series = QBarSeries()
        success = QBarSet("Success Rate")
        success.append([75, 65, 60, 55, 50])
        series.append(success)

        # Add series to chart
        chart.addSeries(series)

        # Set up axes
        categories = ["Shotgun", "I-Form", "Singleback", "Pistol", "Wildcat"]
        axis_x = QBarCategoryAxis()
        axis_x.append(categories)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setRange(0, 100)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        series.attachAxis(axis_y)

        # Create chart view
        chart_view = QChartView(chart)
        chart_view.setRenderHint(chart_view.RenderHint.Antialiasing)
        chart_view.setStyleSheet(
            """
            background: #2A2A2A;
            border-radius: 8px;
        """
        )

        return chart_view
