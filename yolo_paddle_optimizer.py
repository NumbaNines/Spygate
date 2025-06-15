#!/usr/bin/env python3
"""
YOLO + PaddleOCR Preprocessing Optimizer
Uses YOLO model to detect HUD regions, then optimizes preprocessing for each region type
"""

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import matplotlib.pyplot as plt
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image, ImageTk
from ultralytics import YOLO


class YOLOPaddleOptimizer:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO + PaddleOCR Preprocessing Optimizer")
        self.root.geometry("1600x1000")

        # Initialize models
        print("Initializing models...")
        self.init_models()

        # Current image data
        self.original_image = None
        self.processed_image = None
        self.current_image_path = None
        self.yolo_detections = []
        self.selected_region = None

        # Results storage
        self.test_results = {}

        # Create GUI
        self.create_widgets()

        # Load test images
        self.load_test_images()

    def init_models(self):
        """Initialize YOLO and PaddleOCR models"""
        try:
            # Try to load the 8-class model first
            model_paths = [
                "hud_region_training_8class/runs/hud_8class_fp_reduced_speed/weights/best.pt",
                "hud_region_training/runs/detect/train/weights/best.pt",
                "yolov8n.pt",  # Fallback to pretrained
            ]

            self.yolo_model = None
            for model_path in model_paths:
                try:
                    if Path(model_path).exists():
                        self.yolo_model = YOLO(model_path)
                        print(f"✅ Loaded YOLO model: {model_path}")
                        break
                except Exception as e:
                    print(f"❌ Failed to load {model_path}: {e}")
                    continue

            if self.yolo_model is None:
                print("⚠️ No YOLO model found, using basic region detection")

            # Initialize PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
            print("✅ PaddleOCR initialized")

        except Exception as e:
            print(f"❌ Model initialization error: {e}")
            self.yolo_model = None
            self.ocr = None

    def create_widgets(self):
        """Create the main GUI layout"""

        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top panel - Image and detections
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Bottom panel - Controls and results
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X)

        self.create_image_panel(top_frame)
        self.create_control_panel(bottom_frame)

    def create_image_panel(self, parent):
        """Create the image display panel with YOLO detections"""

        # Image controls
        img_controls = ttk.Frame(parent)
        img_controls.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(img_controls, text="Load Image", command=self.load_image).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(img_controls, text="Load Test Folder", command=self.load_test_folder).pack(
            side=tk.LEFT, padx=(0, 5)
        )
        ttk.Button(img_controls, text="Run YOLO Detection", command=self.run_yolo_detection).pack(
            side=tk.LEFT, padx=(0, 5)
        )

        self.image_var = tk.StringVar()
        self.image_combo = ttk.Combobox(
            img_controls, textvariable=self.image_var, state="readonly", width=50
        )
        self.image_combo.pack(side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True)
        self.image_combo.bind("<<ComboboxSelected>>", self.on_image_selected)

        # Image display area
        image_display = ttk.Frame(parent)
        image_display.pack(fill=tk.BOTH, expand=True)

        # Original image with detections
        left_frame = ttk.LabelFrame(
            image_display, text="Original Image + YOLO Detections", padding=10
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.original_label = ttk.Label(left_frame, text="No image loaded")
        self.original_label.pack(fill=tk.BOTH, expand=True)

        # Region selection
        region_frame = ttk.Frame(left_frame)
        region_frame.pack(fill=tk.X, pady=(5, 0))

        ttk.Label(region_frame, text="Select Region:").pack(side=tk.LEFT)
        self.region_var = tk.StringVar()
        self.region_combo = ttk.Combobox(
            region_frame, textvariable=self.region_var, state="readonly"
        )
        self.region_combo.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)
        self.region_combo.bind("<<ComboboxSelected>>", self.on_region_selected)

        # Processed region
        right_frame = ttk.LabelFrame(
            image_display, text="Selected Region + Preprocessing", padding=10
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.processed_label = ttk.Label(right_frame, text="No region selected")
        self.processed_label.pack(fill=tk.BOTH, expand=True)

    def create_control_panel(self, parent):
        """Create the control panel with preprocessing parameters"""

        # Left side - Preprocessing controls
        left_controls = ttk.Frame(parent)
        left_controls.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Right side - Results
        right_controls = ttk.Frame(parent)
        right_controls.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.create_preprocessing_controls(left_controls)
        self.create_results_panel(right_controls)

    def create_preprocessing_controls(self, parent):
        """Create preprocessing parameter controls"""

        preprocess_frame = ttk.LabelFrame(parent, text="Preprocessing Parameters", padding=10)
        preprocess_frame.pack(fill=tk.BOTH, expand=True)

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

        # Upscaling
        scale_frame = ttk.Frame(preprocess_frame)
        scale_frame.pack(fill=tk.X, pady=2)

        ttk.Label(scale_frame, text="Scale Factor:").pack(anchor=tk.W)
        self.scale_factor = tk.DoubleVar(value=2.0)
        scale_scale = ttk.Scale(
            scale_frame,
            from_=1.0,
            to=5.0,
            variable=self.scale_factor,
            orient=tk.HORIZONTAL,
            command=self.update_preview,
        )
        scale_scale.pack(fill=tk.X)
        self.scale_factor_label = ttk.Label(scale_frame, text="2.0")
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

        self.enable_scale = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            enable_frame,
            text="Enable Upscaling",
            variable=self.enable_scale,
            command=self.update_preview,
        ).pack(anchor=tk.W)

        self.enable_gamma = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            enable_frame,
            text="Enable Gamma",
            variable=self.enable_gamma,
            command=self.update_preview,
        ).pack(anchor=tk.W)

        # Action buttons
        action_frame = ttk.Frame(preprocess_frame)
        action_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(action_frame, text="Reset Parameters", command=self.reset_parameters).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(action_frame, text="Run OCR Test", command=self.run_ocr_test).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(action_frame, text="Test All Regions", command=self.test_all_regions).pack(
            fill=tk.X, pady=2
        )
        ttk.Button(action_frame, text="Save Results", command=self.save_results).pack(
            fill=tk.X, pady=2
        )

        # Bind scale updates to labels
        self.contrast_alpha.trace("w", self.update_labels)
        self.contrast_beta.trace("w", self.update_labels)
        self.scale_factor.trace("w", self.update_labels)
        self.gamma.trace("w", self.update_labels)

    def create_results_panel(self, parent):
        """Create the results display panel"""

        results_frame = ttk.LabelFrame(parent, text="OCR Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True)

        # Results text area with scrollbar
        text_frame = ttk.Frame(results_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.results_text = tk.Text(text_frame, height=15, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=scrollbar.set)

        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def update_labels(self, *args):
        """Update parameter labels"""
        self.contrast_alpha_label.config(text=f"{self.contrast_alpha.get():.2f}")
        self.contrast_beta_label.config(text=f"{self.contrast_beta.get():.0f}")
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
                # Use every 10th image for faster testing
                sample_images = image_files[::10][:15]  # Max 15 images
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

            # Clear previous detections
            self.yolo_detections = []
            self.region_combo["values"] = []
            self.selected_region = None

            # Auto-run YOLO detection
            if self.yolo_model:
                self.run_yolo_detection()

        except Exception as e:
            messagebox.showerror("Error", f"Error loading image: {str(e)}")

    def run_yolo_detection(self):
        """Run YOLO detection on current image"""
        if self.original_image is None or self.yolo_model is None:
            return

        try:
            # Run YOLO detection
            results = self.yolo_model(self.original_image, conf=0.3, iou=0.5)

            self.yolo_detections = []
            region_names = []

            for result in results:
                if result.boxes is not None:
                    for i, box in enumerate(result.boxes):
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())

                        # Get class name
                        class_name = (
                            result.names[cls] if cls < len(result.names) else f"class_{cls}"
                        )

                        # Store detection
                        detection = {
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(conf),
                            "class": class_name,
                            "class_id": cls,
                        }
                        self.yolo_detections.append(detection)

                        # Add to region selection
                        region_name = f"{class_name} ({conf:.2f})"
                        region_names.append(region_name)

            # Update region combo
            self.region_combo["values"] = region_names
            if region_names:
                self.region_combo.current(0)
                self.on_region_selected()

            # Display image with detections
            self.display_image_with_detections()

            self.results_text.insert(tk.END, f"🎯 YOLO Detection Results:\n")
            self.results_text.insert(tk.END, f"   Found {len(self.yolo_detections)} regions\n")
            for det in self.yolo_detections:
                self.results_text.insert(tk.END, f"   - {det['class']}: {det['confidence']:.3f}\n")
            self.results_text.insert(tk.END, "\n")
            self.results_text.see(tk.END)

        except Exception as e:
            self.results_text.insert(tk.END, f"❌ YOLO Detection Error: {str(e)}\n")

    def display_image_with_detections(self):
        """Display image with YOLO detection boxes"""
        if self.original_image is None:
            return

        # Create a copy for drawing
        display_image = self.original_image.copy()

        # Draw detection boxes
        for detection in self.yolo_detections:
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]
            class_name = detection["class"]

            # Draw rectangle
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                display_image,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                display_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1
            )

        self.display_image(display_image, self.original_label, "Original + Detections")

    def on_region_selected(self, event=None):
        """Handle region selection"""
        if not self.yolo_detections:
            return

        selected_index = self.region_combo.current()
        if selected_index >= 0 and selected_index < len(self.yolo_detections):
            self.selected_region = self.yolo_detections[selected_index]
            self.update_preview()

    def apply_preprocessing(self, image):
        """Apply all enabled preprocessing techniques"""
        processed = image.copy()

        # Contrast enhancement
        if self.enable_contrast.get():
            alpha = self.contrast_alpha.get()
            beta = self.contrast_beta.get()
            processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)

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
        """Update the processed region preview"""
        if self.original_image is None or self.selected_region is None:
            return

        # Extract region
        x1, y1, x2, y2 = self.selected_region["bbox"]
        region = self.original_image[y1:y2, x1:x2]

        if region.size == 0:
            return

        # Apply preprocessing
        self.processed_image = self.apply_preprocessing(region)
        self.display_image(self.processed_image, self.processed_label, "Processed Region")

    def display_image(self, cv_image, label_widget, title):
        """Display a CV2 image in a tkinter label"""
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Resize for display (max 500x400)
            height, width = rgb_image.shape[:2]
            max_width, max_height = 500, 400

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
        """Run OCR on current processed region"""
        if self.processed_image is None or self.ocr is None:
            messagebox.showwarning("Warning", "No processed region or OCR not available")
            return

        self.results_text.insert(tk.END, "Running OCR test on selected region...\n")
        self.root.update()

        try:
            # Run OCR
            result = self.ocr.ocr(self.processed_image, cls=False)

            # Parse results
            if result and result[0]:
                region_name = self.selected_region["class"]
                self.results_text.insert(tk.END, f"\n✅ OCR Results for {region_name}:\n")
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
                    result_key = f"{self.current_image_path}_{region_name}"
                    self.test_results[result_key] = {
                        "image_path": self.current_image_path,
                        "region": region_name,
                        "bbox": self.selected_region["bbox"],
                        "parameters": params,
                        "text_count": text_count,
                        "avg_confidence": avg_confidence,
                        "texts": [line[1][0] for line in result[0] if line and len(line) >= 2],
                        "timestamp": datetime.now().isoformat(),
                    }

            else:
                self.results_text.insert(
                    tk.END, f"\n❌ No text detected in {self.selected_region['class']}\n"
                )

        except Exception as e:
            self.results_text.insert(tk.END, f"\n❌ OCR Error: {str(e)}\n")

        self.results_text.see(tk.END)

    def test_all_regions(self):
        """Test OCR on all detected regions with current parameters"""
        if not self.yolo_detections or self.ocr is None:
            messagebox.showwarning("Warning", "No regions detected or OCR not available")
            return

        self.results_text.insert(tk.END, f"\nTesting all {len(self.yolo_detections)} regions...\n")
        self.root.update()

        successful_tests = 0
        total_confidence = 0
        total_texts = 0

        for i, detection in enumerate(self.yolo_detections):
            region_name = detection["class"]
            self.results_text.insert(
                tk.END, f"\nProcessing {i+1}/{len(self.yolo_detections)}: {region_name}\n"
            )
            self.root.update()

            try:
                # Extract and process region
                x1, y1, x2, y2 = detection["bbox"]
                region = self.original_image[y1:y2, x1:x2]

                if region.size == 0:
                    continue

                processed = self.apply_preprocessing(region)

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

                        # Store result
                        params = self.get_current_parameters()
                        result_key = f"{self.current_image_path}_{region_name}"
                        self.test_results[result_key] = {
                            "image_path": self.current_image_path,
                            "region": region_name,
                            "bbox": detection["bbox"],
                            "parameters": params,
                            "text_count": len(texts),
                            "avg_confidence": avg_conf,
                            "texts": texts,
                            "timestamp": datetime.now().isoformat(),
                        }
                    else:
                        self.results_text.insert(tk.END, f"  ❌ No valid texts\n")
                else:
                    self.results_text.insert(tk.END, f"  ❌ No detection\n")

            except Exception as e:
                self.results_text.insert(tk.END, f"  ❌ Error: {str(e)}\n")

        # Summary
        if successful_tests > 0:
            overall_avg_confidence = total_confidence / successful_tests
            avg_texts_per_region = total_texts / successful_tests
            success_rate = successful_tests / len(self.yolo_detections)

            self.results_text.insert(tk.END, f"\n📊 OVERALL RESULTS:\n")
            self.results_text.insert(
                tk.END,
                f"   Success rate: {success_rate:.1%} ({successful_tests}/{len(self.yolo_detections)})\n",
            )
            self.results_text.insert(
                tk.END, f"   Average confidence: {overall_avg_confidence:.3f}\n"
            )
            self.results_text.insert(
                tk.END, f"   Average texts per region: {avg_texts_per_region:.1f}\n"
            )

        self.results_text.see(tk.END)

    def get_current_parameters(self):
        """Get current preprocessing parameters"""
        return {
            "contrast_enabled": self.enable_contrast.get(),
            "contrast_alpha": self.contrast_alpha.get(),
            "contrast_beta": self.contrast_beta.get(),
            "scale_enabled": self.enable_scale.get(),
            "scale_factor": self.scale_factor.get(),
            "gamma_enabled": self.enable_gamma.get(),
            "gamma": self.gamma.get(),
        }

    def reset_parameters(self):
        """Reset all parameters to defaults"""
        self.contrast_alpha.set(1.5)
        self.contrast_beta.set(30)
        self.scale_factor.set(2.0)
        self.gamma.set(1.0)

        self.enable_contrast.set(True)
        self.enable_scale.set(True)
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
    app = YOLOPaddleOptimizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
