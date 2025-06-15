#!/usr/bin/env python3
"""
PaddleOCR Preprocessing GUI Optimizer
Interactive GUI for testing and optimizing preprocessing parameters for Madden HUD OCR
"""

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageTk


class PreprocessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PaddleOCR Preprocessing Optimizer")
        self.root.geometry("1400x900")

        # Initialize PaddleOCR
        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)

        # Current image data
        self.original_image = None
        self.processed_image = None
        self.current_image_path = None

        # Results storage
        self.test_results = {}

        # Create GUI
        self.create_widgets()

        # Load test images
        self.load_test_images()

    def create_widgets(self):
        """Create the main GUI layout"""

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Right panel - Images and results
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_control_panel(left_frame)
        self.create_image_panel(right_frame)

    def create_control_panel(self, parent):
        """Create the control panel with preprocessing parameters"""

        # Image selection
        img_frame = ttk.LabelFrame(parent, text="Image Selection", padding=10)
        img_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(img_frame, text="Load Image", command=self.load_image).pack(fill=tk.X, pady=2)
        ttk.Button(img_frame, text="Load Test Folder", command=self.load_test_folder).pack(
            fill=tk.X, pady=2
        )

        self.image_var = tk.StringVar()
        self.image_combo = ttk.Combobox(img_frame, textvariable=self.image_var, state="readonly")
        self.image_combo.pack(fill=tk.X, pady=2)
        self.image_combo.bind("<<ComboboxSelected>>", self.on_image_selected)

        # Preprocessing controls
        preprocess_frame = ttk.LabelFrame(parent, text="Preprocessing Parameters", padding=10)
        preprocess_frame.pack(fill=tk.X, pady=(0, 10))

        # Contrast Enhancement
        contrast_frame = ttk.Frame(preprocess_frame)
        contrast_frame.pack(fill=tk.X, pady=2)

        ttk.Label(contrast_frame, text="Contrast Alpha:").pack(anchor=tk.W)
        self.contrast_alpha = tk.DoubleVar(value=1.5)
        contrast_scale = ttk.Scale(
            contrast_frame,
            from_=0.5,
            to=3.0,
            variable=self.contrast_alpha,
            orient=tk.HORIZONTAL,
            command=self.update_preview,
        )
        contrast_scale.pack(fill=tk.X)
        self.contrast_alpha_label = ttk.Label(contrast_frame, text="1.5")
        self.contrast_alpha_label.pack(anchor=tk.W)

        ttk.Label(contrast_frame, text="Brightness Beta:").pack(anchor=tk.W)
        self.contrast_beta = tk.DoubleVar(value=30)
        beta_scale = ttk.Scale(
            contrast_frame,
            from_=0,
            to=100,
            variable=self.contrast_beta,
            orient=tk.HORIZONTAL,
            command=self.update_preview,
        )
        beta_scale.pack(fill=tk.X)
        self.contrast_beta_label = ttk.Label(contrast_frame, text="30")
        self.contrast_beta_label.pack(anchor=tk.W)

        # Gaussian Blur
        blur_frame = ttk.Frame(preprocess_frame)
        blur_frame.pack(fill=tk.X, pady=2)

        ttk.Label(blur_frame, text="Gaussian Blur Kernel:").pack(anchor=tk.W)
        self.blur_kernel = tk.IntVar(value=1)
        blur_scale = ttk.Scale(
            blur_frame,
            from_=1,
            to=15,
            variable=self.blur_kernel,
            orient=tk.HORIZONTAL,
            command=self.update_preview,
        )
        blur_scale.pack(fill=tk.X)
        self.blur_kernel_label = ttk.Label(blur_frame, text="1")
        self.blur_kernel_label.pack(anchor=tk.W)

        # Sharpening
        sharp_frame = ttk.Frame(preprocess_frame)
        sharp_frame.pack(fill=tk.X, pady=2)

        ttk.Label(sharp_frame, text="Sharpening Strength:").pack(anchor=tk.W)
        self.sharp_strength = tk.DoubleVar(value=0.0)
        sharp_scale = ttk.Scale(
            sharp_frame,
            from_=0.0,
            to=2.0,
            variable=self.sharp_strength,
            orient=tk.HORIZONTAL,
            command=self.update_preview,
        )
        sharp_scale.pack(fill=tk.X)
        self.sharp_strength_label = ttk.Label(sharp_frame, text="0.0")
        self.sharp_strength_label.pack(anchor=tk.W)

        # Upscaling
        scale_frame = ttk.Frame(preprocess_frame)
        scale_frame.pack(fill=tk.X, pady=2)

        ttk.Label(scale_frame, text="Scale Factor:").pack(anchor=tk.W)
        self.scale_factor = tk.DoubleVar(value=1.0)
        scale_scale = ttk.Scale(
            scale_frame,
            from_=1.0,
            to=4.0,
            variable=self.scale_factor,
            orient=tk.HORIZONTAL,
            command=self.update_preview,
        )
        scale_scale.pack(fill=tk.X)
        self.scale_factor_label = ttk.Label(scale_frame, text="1.0")
        self.scale_factor_label.pack(anchor=tk.W)

        # Gamma Correction
        gamma_frame = ttk.Frame(preprocess_frame)
        gamma_frame.pack(fill=tk.X, pady=2)

        ttk.Label(gamma_frame, text="Gamma:").pack(anchor=tk.W)
        self.gamma = tk.DoubleVar(value=1.0)
        gamma_scale = ttk.Scale(
            gamma_frame,
            from_=0.3,
            to=3.0,
            variable=self.gamma,
            orient=tk.HORIZONTAL,
            command=self.update_preview,
        )
        gamma_scale.pack(fill=tk.X)
        self.gamma_label = ttk.Label(gamma_frame, text="1.0")
        self.gamma_label.pack(anchor=tk.W)

        # Enable/Disable checkboxes
        enable_frame = ttk.Frame(preprocess_frame)
        enable_frame.pack(fill=tk.X, pady=5)

        self.enable_contrast = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            enable_frame,
            text="Enable Contrast",
            variable=self.enable_contrast,
            command=self.update_preview,
        ).pack(anchor=tk.W)

        self.enable_blur = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            enable_frame, text="Enable Blur", variable=self.enable_blur, command=self.update_preview
        ).pack(anchor=tk.W)

        self.enable_sharp = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            enable_frame,
            text="Enable Sharpening",
            variable=self.enable_sharp,
            command=self.update_preview,
        ).pack(anchor=tk.W)

        self.enable_scale = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            enable_frame,
            text="Enable Upscaling",
            variable=self.enable_scale,
            command=self.update_preview,
        ).pack(anchor=tk.X)

        self.enable_gamma = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            enable_frame,
            text="Enable Gamma",
            variable=self.enable_gamma,
            command=self.update_preview,
        ).pack(anchor=tk.W)

        # Action buttons
        action_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        action_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(action_frame, text="Reset Parameters", command=self.reset_parameters).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(action_frame, text="Run OCR Test", command=self.run_ocr_test).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(action_frame, text="Test All Images", command=self.test_all_images).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(action_frame, text="Save Results", command=self.save_results).pack(
            fill=tk.X, pady=2
        )

        # Bind scale updates to labels
        self.contrast_alpha.trace("w", self.update_labels)
        self.contrast_beta.trace("w", self.update_labels)
        self.blur_kernel.trace("w", self.update_labels)
        self.sharp_strength.trace("w", self.update_labels)
        self.scale_factor.trace("w", self.update_labels)
        self.gamma.trace("w", self.update_labels)

    def create_image_panel(self, parent):
        """Create the image display and results panel"""

        # Image display
        image_frame = ttk.LabelFrame(parent, text="Image Comparison", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Original and processed image labels
        img_container = ttk.Frame(image_frame)
        img_container.pack(fill=tk.BOTH, expand=True)

        # Original image
        orig_frame = ttk.Frame(img_container)
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        ttk.Label(orig_frame, text="Original Image").pack()
        self.original_label = ttk.Label(orig_frame, text="No image loaded")
        self.original_label.pack(fill=tk.BOTH, expand=True)

        # Processed image
        proc_frame = ttk.Frame(img_container)
        proc_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        ttk.Label(proc_frame, text="Processed Image").pack()
        self.processed_label = ttk.Label(proc_frame, text="No image loaded")
        self.processed_label.pack(fill=tk.BOTH, expand=True)

        # Results panel
        results_frame = ttk.LabelFrame(parent, text="OCR Results", padding=10)
        results_frame.pack(fill=tk.X)

        # Results text area with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(text_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def update_labels(self, *args):
        """Update parameter labels"""
        self.contrast_alpha_label.config(text=f"{self.contrast_alpha.get():.2f}")
        self.contrast_beta_label.config(text=f"{self.contrast_beta.get():.0f}")
        self.blur_kernel_label.config(text=f"{self.blur_kernel.get()}")
        self.sharp_strength_label.config(text=f"{self.sharp_strength.get():.2f}")
        self.scale_factor_label.config(text=f"{self.scale_factor.get():.2f}")
        self.gamma_label.config(text=f"{self.gamma.get():.2f}")

    def load_image(self):
        """Load a single image"""
        file_path = filedialog.askopenfilename(
            title="Select Image", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )

        if file_path:
            self.current_image_path = file_path
            self.load_and_display_image(file_path)

    def load_test_folder(self):
        """Load all images from a folder"""
        folder_path = filedialog.askdirectory(title="Select Test Images Folder")

        if folder_path:
            folder = Path(folder_path)
            image_files = []

            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff"]:
                image_files.extend(folder.glob(ext))

            if image_files:
                self.image_combo["values"] = [str(f) for f in image_files]
                self.image_combo.current(0)
                self.on_image_selected()
                messagebox.showinfo("Success", f"Loaded {len(image_files)} images")
            else:
                messagebox.showwarning("Warning", "No image files found in selected folder")

    def load_test_images(self):
        """Load default test images from 6.12 screenshots"""
        screenshots_folder = Path("6.12 screenshots")
        if screenshots_folder.exists():
            image_files = list(screenshots_folder.glob("*.png"))
            if image_files:
                # Use every 5th image for faster testing
                sample_images = image_files[::5][:20]  # Max 20 images
                self.image_combo["values"] = [str(f) for f in sample_images]
                if sample_images:
                    self.image_combo.current(0)
                    self.on_image_selected()

    def on_image_selected(self, event=None):
        """Handle image selection from combobox"""
        selected_path = self.image_var.get()
        if selected_path:
            self.current_image_path = selected_path
            self.load_and_display_image(selected_path)

    def load_and_display_image(self, image_path):
        """Load and display an image"""
        try:
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                messagebox.showerror("Error", f"Could not load image: {image_path}")
                return

            # Display original image
            self.display_image(self.original_image, self.original_label, "Original")

            # Update processed image
            self.update_preview()

        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def apply_preprocessing(self, image):
        """Apply all enabled preprocessing techniques"""
        processed = image.copy()

        # Contrast enhancement
        if self.enable_contrast.get():
            alpha = self.contrast_alpha.get()
            beta = self.contrast_beta.get()
            processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)

        # Gaussian blur
        if self.enable_blur.get():
            kernel = self.blur_kernel.get()
            if kernel > 1:
                # Ensure odd kernel size
                if kernel % 2 == 0:
                    kernel += 1
                processed = cv2.GaussianBlur(processed, (kernel, kernel), 0)

        # Sharpening
        if self.enable_sharp.get():
            strength = self.sharp_strength.get()
            if strength > 0:
                gaussian = cv2.GaussianBlur(processed, (0, 0), 2.0)
                processed = cv2.addWeighted(processed, 1.0 + strength, gaussian, -strength, 0)

        # Upscaling
        if self.enable_scale.get():
            scale = self.scale_factor.get()
            if scale > 1.0:
                height, width = processed.shape[:2]
                new_width = int(width * scale)
                new_height = int(height * scale)
                processed = cv2.resize(
                    processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC
                )

        # Gamma correction
        if self.enable_gamma.get():
            gamma = self.gamma.get()
            if gamma != 1.0:
                inv_gamma = 1.0 / gamma
                table = np.array(
                    [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]
                ).astype("uint8")
                processed = cv2.LUT(processed, table)

        return processed

    def update_preview(self, *args):
        """Update the processed image preview"""
        if self.original_image is not None:
            self.processed_image = self.apply_preprocessing(self.original_image)
            self.display_image(self.processed_image, self.processed_label, "Processed")

    def display_image(self, cv_image, label_widget, title):
        """Display a CV2 image in a tkinter label"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Resize for display (max 400x300)
            height, width = rgb_image.shape[:2]
            max_width, max_height = 400, 300

            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                rgb_image = cv2.resize(rgb_image, (new_width, new_height))

            # Convert to PIL and then to PhotoImage
            pil_image = Image.fromarray(rgb_image)
            photo = ImageTk.PhotoImage(pil_image)

            # Update label
            label_widget.configure(image=photo, text="")
            label_widget.image = photo  # Keep a reference

        except Exception as e:
            label_widget.configure(text=f"Error displaying {title}: {str(e)}")

    def run_ocr_test(self):
        """Run OCR on current processed image"""
        if self.processed_image is None:
            messagebox.showwarning("Warning", "No image loaded")
            return

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Running OCR test...\n")
        self.root.update()

        try:
            # Run OCR
            result = self.ocr.ocr(self.processed_image, cls=False)

            # Parse results
            if result and result[0]:
                self.results_text.insert(
                    tk.END, f"\n✅ OCR Results for {Path(self.current_image_path).name}:\n"
                )
                self.results_text.insert(tk.END, "=" * 50 + "\n")

                total_confidence = 0
                text_count = 0

                for i, line in enumerate(result[0]):
                    if line and len(line) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        total_confidence += confidence
                        text_count += 1

                        self.results_text.insert(
                            tk.END, f"{i+1:2d}. '{text}' (confidence: {confidence:.3f})\n"
                        )

                if text_count > 0:
                    avg_confidence = total_confidence / text_count
                    self.results_text.insert(tk.END, f"\n📊 Summary:\n")
                    self.results_text.insert(tk.END, f"   Texts detected: {text_count}\n")
                    self.results_text.insert(
                        tk.END, f"   Average confidence: {avg_confidence:.3f}\n"
                    )

                    # Store result
                    params = self.get_current_parameters()
                    self.test_results[self.current_image_path] = {
                        "parameters": params,
                        "text_count": text_count,
                        "avg_confidence": avg_confidence,
                        "texts": [line[1][0] for line in result[0] if line and len(line) >= 2],
                        "timestamp": datetime.now().isoformat(),
                    }

            else:
                self.results_text.insert(tk.END, "\n❌ No text detected\n")

        except Exception as e:
            self.results_text.insert(tk.END, f"\n❌ OCR Error: {str(e)}\n")

        self.results_text.see(tk.END)

    def test_all_images(self):
        """Test OCR on all loaded images with current parameters"""
        if not self.image_combo["values"]:
            messagebox.showwarning("Warning", "No images loaded")
            return

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Testing all images...\n")
        self.root.update()

        total_images = len(self.image_combo["values"])
        successful_tests = 0
        total_confidence = 0
        total_texts = 0

        for i, image_path in enumerate(self.image_combo["values"]):
            self.results_text.insert(
                tk.END, f"\nProcessing {i+1}/{total_images}: {Path(image_path).name}\n"
            )
            self.root.update()

            try:
                # Load and process image
                image = cv2.imread(image_path)
                if image is None:
                    continue

                processed = self.apply_preprocessing(image)

                # Run OCR
                result = self.ocr.ocr(processed, cls=False)

                if result and result[0]:
                    texts = [line[1][0] for line in result[0] if line and len(line) >= 2]
                    confidences = [line[1][1] for line in result[0] if line and len(line) >= 2]

                    if texts:
                        avg_conf = np.mean(confidences)
                        successful_tests += 1
                        total_confidence += avg_conf
                        total_texts += len(texts)

                        self.results_text.insert(
                            tk.END, f"  ✅ {len(texts)} texts, avg confidence: {avg_conf:.3f}\n"
                        )
                    else:
                        self.results_text.insert(tk.END, f"  ❌ No valid texts\n")
                else:
                    self.results_text.insert(tk.END, f"  ❌ No detection\n")

            except Exception as e:
                self.results_text.insert(tk.END, f"  ❌ Error: {str(e)}\n")

        # Summary
        if successful_tests > 0:
            overall_avg_confidence = total_confidence / successful_tests
            avg_texts_per_image = total_texts / successful_tests
            success_rate = successful_tests / total_images

            self.results_text.insert(tk.END, f"\n📊 OVERALL RESULTS:\n")
            self.results_text.insert(
                tk.END, f"   Success rate: {success_rate:.1%} ({successful_tests}/{total_images})\n"
            )
            self.results_text.insert(
                tk.END, f"   Average confidence: {overall_avg_confidence:.3f}\n"
            )
            self.results_text.insert(
                tk.END, f"   Average texts per image: {avg_texts_per_image:.1f}\n"
            )

        self.results_text.see(tk.END)

    def get_current_parameters(self):
        """Get current preprocessing parameters"""
        return {
            "contrast_enabled": self.enable_contrast.get(),
            "contrast_alpha": self.contrast_alpha.get(),
            "contrast_beta": self.contrast_beta.get(),
            "blur_enabled": self.enable_blur.get(),
            "blur_kernel": self.blur_kernel.get(),
            "sharp_enabled": self.enable_sharp.get(),
            "sharp_strength": self.sharp_strength.get(),
            "scale_enabled": self.enable_scale.get(),
            "scale_factor": self.scale_factor.get(),
            "gamma_enabled": self.enable_gamma.get(),
            "gamma": self.gamma.get(),
        }

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self.contrast_alpha.set(1.5)
        self.contrast_beta.set(30)
        self.blur_kernel.set(1)
        self.sharp_strength.set(0.0)
        self.scale_factor.set(1.0)
        self.gamma.set(1.0)

        self.enable_contrast.set(True)
        self.enable_blur.set(False)
        self.enable_sharp.set(False)
        self.enable_scale.set(False)
        self.enable_gamma.set(False)

        self.update_preview()

    def save_results(self):
        """Save test results to JSON file"""
        if not self.test_results:
            messagebox.showwarning("Warning", "No test results to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Results", defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )

        if file_path:
            try:
                with open(file_path, "w") as f:
                    json.dump(self.test_results, f, indent=2)
                messagebox.showinfo("Success", f"Results saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Error saving results: {str(e)}")


def main():
    root = tk.Tk()
    app = PreprocessingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
