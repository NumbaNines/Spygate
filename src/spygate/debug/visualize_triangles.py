import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_triangle_detection(image_path: str, save_path: str):
    """
    Create a visual breakdown of how we detect triangles in the HUD.
    Focuses on the bottom HUD area where triangles appear.
    """
    # Read the image
    frame = cv2.imread(image_path)

    # Extract the HUD region (bottom 10% of screen where triangles appear)
    height = frame.shape[0]
    hud_region = frame[int(height * 0.9) :, :]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Triangle Detection Process", fontsize=16)

    # 1. Original HUD region
    axes[0, 0].imshow(cv2.cvtColor(hud_region, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("1. HUD Region")

    # 2. Convert to grayscale
    gray = cv2.cvtColor(hud_region, cv2.COLOR_BGR2GRAY)
    axes[0, 1].imshow(gray, cmap="gray")
    axes[0, 1].set_title("2. Grayscale")

    # 3. Binary threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    axes[0, 2].imshow(thresh, cmap="gray")
    axes[0, 2].set_title("3. Binary Threshold")

    # 4. Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw all contours
    contour_img = np.zeros_like(thresh)
    cv2.drawContours(contour_img, contours, -1, 255, 2)
    axes[1, 0].imshow(contour_img, cmap="gray")
    axes[1, 0].set_title("4. All Contours")

    # 5. Filter triangle-like contours
    triangle_img = hud_region.copy()
    for cnt in contours:
        # Simple triangle detection
        if len(cnt) >= 3:  # Must have at least 3 points
            area = cv2.contourArea(cnt)
            if 100 < area < 1000:  # Reasonable triangle size
                cv2.drawContours(triangle_img, [cnt], -1, (0, 255, 0), 2)

                # Draw extreme points
                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
                bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])

                # Draw points in different colors
                cv2.circle(triangle_img, leftmost, 3, (255, 0, 0), -1)  # Blue
                cv2.circle(triangle_img, rightmost, 3, (0, 0, 255), -1)  # Red
                cv2.circle(triangle_img, topmost, 3, (255, 255, 0), -1)  # Yellow
                cv2.circle(triangle_img, bottommost, 3, (0, 255, 255), -1)  # Cyan

    axes[1, 1].imshow(cv2.cvtColor(triangle_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("5. Triangle Detection")

    # 6. Final analysis overlay
    final = hud_region.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Add analysis text
    cv2.putText(final, "DND vs NBA", (10, 30), font, 0.7, (255, 255, 255), 2)
    cv2.putText(final, "Score: 23-16", (10, 60), font, 0.7, (255, 255, 255), 2)

    axes[1, 2].imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title("6. Final Analysis")

    # Remove axes ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    visualize_triangle_detection("screenshot.png", "triangle_analysis.png")
