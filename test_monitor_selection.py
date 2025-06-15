#!/usr/bin/env python3
"""
Test Monitor Selection for SpygateAI
====================================
Simple script to test monitor detection and selection functionality.
"""

import sys
import time

import cv2
import numpy as np

try:
    import mss

    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    print("❌ MSS not available. Install with: pip install mss")
    sys.exit(1)


def list_monitors():
    """List all available monitors."""
    print("🖥️  Detecting Available Monitors...")
    print("=" * 50)

    try:
        with mss.mss() as sct:
            monitors = sct.monitors

            print(f"Total monitors detected: {len(monitors)}")
            print("\nMonitor Details:")
            print("-" * 50)

            # Skip index 0 (all monitors combined)
            for i, monitor in enumerate(monitors):
                if i == 0:
                    print(
                        f"Monitor 0: All Monitors Combined - {monitor['width']}x{monitor['height']}"
                    )
                else:
                    print(
                        f"Monitor {i}: {monitor['width']}x{monitor['height']} "
                        f"at position ({monitor['left']}, {monitor['top']})"
                    )

            return monitors

    except Exception as e:
        print(f"❌ Error detecting monitors: {e}")
        return None


def capture_monitor_screenshot(monitor_num, monitors):
    """Capture a screenshot from the specified monitor."""
    try:
        with mss.mss() as sct:
            if monitor_num >= len(monitors):
                print(f"❌ Monitor {monitor_num} not found!")
                return None

            monitor = monitors[monitor_num]
            print(f"\n📸 Capturing screenshot from Monitor {monitor_num}...")
            print(f"   Resolution: {monitor['width']}x{monitor['height']}")
            print(f"   Position: ({monitor['left']}, {monitor['top']})")

            # Capture screenshot
            screenshot = sct.grab(monitor)

            # Convert to numpy array
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            return img

    except Exception as e:
        print(f"❌ Error capturing screenshot: {e}")
        return None


def show_monitor_preview(monitor_num, monitors, duration=3):
    """Show a live preview of the specified monitor."""
    try:
        with mss.mss() as sct:
            if monitor_num >= len(monitors):
                print(f"❌ Monitor {monitor_num} not found!")
                return

            monitor = monitors[monitor_num]
            print(f"\n📺 Showing live preview of Monitor {monitor_num} for {duration} seconds...")
            print("Press 'q' to quit early or wait for auto-close")

            start_time = time.time()

            while time.time() - start_time < duration:
                # Capture frame
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                # Resize if too large
                height, width = frame.shape[:2]
                if width > 1280 or height > 720:
                    scale = min(1280 / width, 720 / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv2.resize(frame, (new_width, new_height))

                # Add overlay text
                cv2.putText(
                    frame,
                    f"Monitor {monitor_num} Preview",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Resolution: {monitor['width']}x{monitor['height']}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Press 'q' to quit",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                # Show frame
                cv2.imshow(f"Monitor {monitor_num} Preview", frame)

                # Check for quit
                if cv2.waitKey(100) & 0xFF == ord("q"):
                    break

            cv2.destroyAllWindows()
            print("✅ Preview complete!")

    except Exception as e:
        print(f"❌ Error showing preview: {e}")


def interactive_monitor_selection():
    """Interactive monitor selection with preview."""
    monitors = list_monitors()
    if not monitors:
        return

    print("\n" + "=" * 50)

    while True:
        try:
            choice = (
                input(f"\nSelect monitor (1-{len(monitors)-1}), 'p' for preview, or 'q' to quit: ")
                .strip()
                .lower()
            )

            if choice == "q":
                print("👋 Goodbye!")
                break
            elif choice == "p":
                try:
                    monitor_num = int(input("Which monitor to preview? "))
                    if 1 <= monitor_num <= len(monitors) - 1:
                        show_monitor_preview(monitor_num, monitors)
                    else:
                        print(f"❌ Please enter a number between 1 and {len(monitors)-1}")
                except ValueError:
                    print("❌ Please enter a valid number")
            else:
                try:
                    monitor_num = int(choice)
                    if 1 <= monitor_num <= len(monitors) - 1:
                        selected_monitor = monitors[monitor_num]
                        print(f"\n✅ Selected Monitor {monitor_num}:")
                        print(
                            f"   Resolution: {selected_monitor['width']}x{selected_monitor['height']}"
                        )
                        print(
                            f"   Position: ({selected_monitor['left']}, {selected_monitor['top']})"
                        )

                        # Ask for preview
                        preview = input("Show preview? (y/n): ").strip().lower()
                        if preview in ["y", "yes"]:
                            show_monitor_preview(monitor_num, monitors)

                        # Ask for screenshot
                        screenshot = input("Save screenshot? (y/n): ").strip().lower()
                        if screenshot in ["y", "yes"]:
                            img = capture_monitor_screenshot(monitor_num, monitors)
                            if img is not None:
                                filename = f"monitor_{monitor_num}_screenshot.png"
                                cv2.imwrite(filename, img)
                                print(f"✅ Screenshot saved as {filename}")

                        break
                    else:
                        print(f"❌ Please enter a number between 1 and {len(monitors)-1}")
                except ValueError:
                    print("❌ Please enter a valid number, 'p' for preview, or 'q' to quit")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


def main():
    """Main function."""
    print("🏈 SpygateAI Monitor Selection Test")
    print("=" * 50)

    if not MSS_AVAILABLE:
        print("❌ This test requires the 'mss' library")
        print("Install with: pip install mss")
        return

    interactive_monitor_selection()


if __name__ == "__main__":
    main()
