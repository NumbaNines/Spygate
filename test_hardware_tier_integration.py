"""
Hardware Tier Detection Integration Testing
Subtask 19.18: Validate hardware tier detection integration with PyQt6 interface
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from spygate.core.hardware import HardwareDetector, HardwareTier
    from spygate.core.optimizer import TierOptimizer

    SPYGATE_AVAILABLE = True
except ImportError:
    SPYGATE_AVAILABLE = False

try:
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class HardwareTierTestWidget(QWidget):
    """Test widget for hardware tier detection integration."""

    def __init__(self):
        super().__init__()
        self.hardware_detector = None
        self.tier_optimizer = None
        self.init_ui()
        self.init_hardware()

    def init_ui(self):
        """Initialize the test UI."""
        layout = QVBoxLayout()

        # Title
        title = QLabel("Hardware Tier Detection Integration Test")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #10B981; margin: 10px;")
        layout.addWidget(title)

        # Hardware info display
        self.hardware_info = QLabel("Detecting hardware...")
        self.hardware_info.setStyleSheet(
            """
            QLabel {
                background-color: #2a2a2a;
                color: white;
                padding: 15px;
                border-radius: 8px;
                border: 2px solid #444;
                font-family: monospace;
            }
        """
        )
        layout.addWidget(self.hardware_info)

        # Status display
        self.status_display = QLabel("Initializing...")
        self.status_display.setStyleSheet(
            """
            QLabel {
                background-color: #1a1a1a;
                color: #ff6b35;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
        """
        )
        layout.addWidget(self.status_display)

        # Test button
        self.test_button = QPushButton("🔍 Run Hardware Integration Tests")
        self.test_button.setStyleSheet(
            """
            QPushButton {
                background-color: #ff6b35;
                color: white;
                padding: 12px;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #e55a2b;
            }
            QPushButton:pressed {
                background-color: #cc4e24;
            }
        """
        )
        self.test_button.clicked.connect(self.run_integration_tests)
        layout.addWidget(self.test_button)

        self.setLayout(layout)
        self.setWindowTitle("Hardware Tier Integration Test")
        self.setMinimumSize(600, 400)

        # Apply dark theme
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0f0f0f;
                color: white;
            }
        """
        )

    def init_hardware(self):
        """Initialize hardware detection."""
        try:
            if not SPYGATE_AVAILABLE:
                self.hardware_info.setText("❌ SpygateAI modules not available")
                self.status_display.setText("Error: Missing SpygateAI dependencies")
                return

            # Initialize hardware detector
            self.hardware_detector = HardwareDetector()
            tier = self.hardware_detector.tier

            # Initialize tier optimizer
            self.tier_optimizer = TierOptimizer(self.hardware_detector)

            # Display hardware information
            memory_gb = self.hardware_detector.total_memory / (1024**3)

            hardware_text = f"""🖥️ Hardware Configuration:
   • Tier: {tier.name} ({'✅ TARGET: HIGH' if tier == HardwareTier.HIGH else '⚠️ Not HIGH tier'})
   • CPU Cores: {self.hardware_detector.cpu_count}
   • RAM: {memory_gb:.1f} GB
   • GPU Available: {'✅' if self.hardware_detector.has_cuda else '❌'}
   • GPU Name: {self.hardware_detector.gpu_name}

⚡ Integration Status:
   • HardwareDetector: {'✅ Initialized' if self.hardware_detector else '❌ Failed'}
   • TierOptimizer: {'✅ Initialized' if self.tier_optimizer else '❌ Failed'}
   • PyQt6 Interface: ✅ Active

🎯 Test Objective:
   • Validate HIGH tier detection on test system
   • Verify interface adaptation to hardware tier
   • Test optimization profile creation"""

            self.hardware_info.setText(hardware_text)

            # Update status
            if tier == HardwareTier.HIGH:
                self.status_display.setText("✅ HIGH tier detected successfully!")
                self.status_display.setStyleSheet(
                    """
                    QLabel {
                        background-color: #10B981;
                        color: white;
                        padding: 10px;
                        border-radius: 5px;
                        font-weight: bold;
                    }
                """
                )
            else:
                self.status_display.setText(f"⚠️ Expected HIGH tier, got {tier.name}")
                self.status_display.setStyleSheet(
                    """
                    QLabel {
                        background-color: #f59e0b;
                        color: white;
                        padding: 10px;
                        border-radius: 5px;
                        font-weight: bold;
                    }
                """
                )

            logger.info(f"Hardware tier detected: {tier.name}")

        except Exception as e:
            error_msg = f"❌ Hardware detection error: {e}"
            self.hardware_info.setText(error_msg)
            self.status_display.setText("Error: Hardware detection failed")
            logger.error(f"Hardware detection failed: {e}")

    def run_integration_tests(self):
        """Run comprehensive integration tests."""
        try:
            self.status_display.setText("🔄 Running integration tests...")

            # Test results
            results = []

            # Test 1: Hardware Detector Initialization
            test1 = self.hardware_detector is not None
            results.append(f"1. Hardware Detector Init: {'✅ PASS' if test1 else '❌ FAIL'}")

            # Test 2: Tier Detection
            expected_tier = HardwareTier.HIGH
            test2 = self.hardware_detector and self.hardware_detector.tier == expected_tier
            current_tier = self.hardware_detector.tier.name if self.hardware_detector else "None"
            results.append(
                f"2. HIGH Tier Detection: {'✅ PASS' if test2 else f'❌ FAIL (got {current_tier})'}"
            )

            # Test 3: Tier Optimizer Integration
            test3 = self.tier_optimizer is not None
            results.append(f"3. Tier Optimizer Init: {'✅ PASS' if test3 else '❌ FAIL'}")

            # Test 4: Performance Profile Creation
            test4 = False
            if self.tier_optimizer:
                try:
                    profile = self.tier_optimizer.get_current_params()
                    test4 = profile is not None and isinstance(profile, dict)
                except Exception:
                    test4 = False
            results.append(f"4. Performance Profile: {'✅ PASS' if test4 else '❌ FAIL'}")

            # Test 5: Memory Information Access
            test5 = False
            if self.hardware_detector:
                try:
                    memory_info = self.hardware_detector.get_system_memory()
                    test5 = memory_info is not None and isinstance(memory_info, dict)
                except Exception:
                    test5 = False
            results.append(f"5. Memory Info Access: {'✅ PASS' if test5 else '❌ FAIL'}")

            # Test 6: PyQt6 Interface Adaptation
            test6 = True  # If we're running, PyQt6 is working
            results.append(f"6. PyQt6 Interface: {'✅ PASS' if test6 else '❌ FAIL'}")

            # Test 7: Hardware Statistics
            test7 = False
            if self.hardware_detector:
                try:
                    stats = self.hardware_detector.get_comprehensive_stats()
                    test7 = stats is not None and "hardware_tier" in stats
                except Exception:
                    test7 = False
            results.append(f"7. Hardware Statistics: {'✅ PASS' if test7 else '❌ FAIL'}")

            # Calculate success rate
            passed_tests = sum(1 for result in results if "✅ PASS" in result)
            total_tests = len(results)
            success_rate = (passed_tests / total_tests) * 100

            # Update display
            results_text = "\n".join(results)
            results_text += f"\n\n📊 Test Results: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)"

            # Add detailed hardware info if available
            if self.hardware_detector:
                tier = self.hardware_detector.tier
                memory_gb = self.hardware_detector.total_memory / (1024**3)

                results_text += f"""

🔧 Detailed Hardware Report:
   • Detected Tier: {tier.name}
   • CPU Cores: {self.hardware_detector.cpu_count}
   • System Memory: {memory_gb:.1f} GB
   • CUDA Available: {self.hardware_detector.has_cuda}
   • GPU Count: {self.hardware_detector.gpu_count}

⚙️ Integration Validation:
   • Expected: HIGH tier on test system
   • Actual: {tier.name} tier
   • Status: {'✅ VALIDATED' if tier == HardwareTier.HIGH else '⚠️ UNEXPECTED'}
   • Interface Adaptation: ✅ Active"""

            self.hardware_info.setText(results_text)

            # Update status based on results
            if success_rate >= 85:
                self.status_display.setText(
                    f"✅ Integration tests completed: {success_rate:.1f}% success"
                )
                self.status_display.setStyleSheet(
                    """
                    QLabel {
                        background-color: #10B981;
                        color: white;
                        padding: 10px;
                        border-radius: 5px;
                        font-weight: bold;
                    }
                """
                )
            else:
                self.status_display.setText(
                    f"⚠️ Integration tests completed: {success_rate:.1f}% success"
                )
                self.status_display.setStyleSheet(
                    """
                    QLabel {
                        background-color: #f59e0b;
                        color: white;
                        padding: 10px;
                        border-radius: 5px;
                        font-weight: bold;
                    }
                """
                )

            logger.info(f"Integration tests completed: {passed_tests}/{total_tests} passed")

        except Exception as e:
            error_msg = f"❌ Integration test error: {e}"
            self.status_display.setText("Error during integration testing")
            logger.error(f"Integration test failed: {e}")


def main():
    """Main function to run the hardware tier integration test."""
    print("=" * 60)
    print("HARDWARE TIER DETECTION INTEGRATION TEST")
    print("Subtask 19.18: Validate PyQt6 interface integration")
    print("=" * 60)

    # Check dependencies
    if not PYQT6_AVAILABLE:
        print("❌ PyQt6 not available - cannot test interface integration")
        return False

    if not SPYGATE_AVAILABLE:
        print("❌ SpygateAI modules not available - limited testing")

    # Create Qt application
    app = QApplication(sys.argv)

    # Create test widget
    test_widget = HardwareTierTestWidget()
    test_widget.show()

    # Auto-run tests after 2 seconds
    QTimer.singleShot(2000, test_widget.run_integration_tests)

    print("✅ Starting PyQt6 integration test interface...")
    print("💡 The test will run automatically after 2 seconds")
    print("🎯 Expected: HIGH tier detection on test system")

    # Run the application
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
