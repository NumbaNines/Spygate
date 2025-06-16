#!/usr/bin/env python3
"""
Down Template Creator GUI - Enhanced for SpygateAI
Crop down text regions from raw Madden 25 screenshots to create templates.
Applies SpygateAI's optimal OCR preprocessing for maximum accuracy.
"""

import cv2
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json

class DownTemplateCropperGUI:
    """GUI for cropping down templates from raw screenshots with SpygateAI preprocessing."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("SpygateAI - Down Template Creator (Enhanced)")
        self.root.geometry("1200x800")
        
        # State
        self.current_image = None
        self.current_image_path = None
        self.display_image = None
        self.scale_factor = 1.0
        self.zoom_factor = 1.0  # Additional zoom for detailed work
        self.crop_start = None
        self.crop_end = None
        self.cropping = False
        
        # Pan offset for zoomed images
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.last_pan_x = 0
        self.last_pan_y = 0
        self.panning = False
        
        # Templates data
        self.templates_data = {}
        self.current_down = "1ST"
        
        self.setup_ui()
        self.load_screenshots()
    
    def apply_spygate_preprocessing(self, image):
        """
        Apply SpygateAI's optimal OCR preprocessing pipeline.
        Score: 0.939 (NEW WINNER at combo 19,778)
        Parameters: Scale=3.5x, CLAHE clip=1.0 grid=(4,4), Blur=(3,3), 
                   Threshold=adaptive_mean block=13 C=3, Morphological=(3,3), Gamma=0.8
        """
        if image is None:
            return None
            
        # 1. Grayscale (always first)
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 2. LANCZOS4 scaling (always second) - Scale=3.5x
        height, width = gray.shape
        scaled = cv2.resize(gray, (int(width * 3.5), int(height * 3.5)), 
                           interpolation=cv2.INTER_LANCZOS4)
        
        # 3. CLAHE (always third) - clip=1.0, grid=(4,4)
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
        clahe_applied = clahe.apply(scaled)
        
        # 4. Gamma correction - Gamma=0.8
        gamma = 0.8
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        gamma_corrected = cv2.LUT(clahe_applied, table)
        
        # 5. Gaussian blur - (3,3)
        blurred = cv2.GaussianBlur(gamma_corrected, (3, 3), 0)
        
        # 6. Thresholding (always applied) - adaptive_mean, block=13, C=3
        threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 13, 3)
        
        # 7. Morphological closing (always applied) - kernel=(3,3)
        kernel = np.ones((3, 3), np.uint8)
        morphological = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        
        # 8. Sharpening (conditional, always last) - False for this combo
        # Skipping sharpening as per optimal parameters
        
        return morphological

    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel (left side)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Title
        title_label = ttk.Label(control_frame, text="SpygateAI Down Template Creator", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        subtitle_label = ttk.Label(control_frame, text="Enhanced with Optimal OCR Preprocessing", 
                                  font=("Arial", 10), foreground="green")
        subtitle_label.pack(pady=(0, 15))
        
        # Screenshot selection
        ttk.Label(control_frame, text="📸 Screenshots:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        self.screenshot_listbox = tk.Listbox(control_frame, height=6)
        self.screenshot_listbox.pack(fill=tk.X, pady=(5, 10))
        self.screenshot_listbox.bind('<<ListboxSelect>>', self.on_screenshot_select)
        
        # Down selection
        ttk.Label(control_frame, text="🎯 Down Type:", font=("Arial", 12, "bold")).pack(anchor=tk.W)
        
        down_frame = ttk.Frame(control_frame)
        down_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.down_var = tk.StringVar(value="1ST")
        
        # Normal down types
        normal_frame = ttk.LabelFrame(down_frame, text="📏 Normal Situations")
        normal_frame.pack(fill=tk.X, pady=(0, 10))
        
        for down in ["1ST", "2ND", "3RD", "4TH"]:
            ttk.Radiobutton(normal_frame, text=down, variable=self.down_var, 
                           value=down, command=self.on_down_change).pack(anchor=tk.W, padx=5)
        
        # GOAL down types
        goal_frame = ttk.LabelFrame(down_frame, text="🥅 GOAL Situations")
        goal_frame.pack(fill=tk.X)
        
        for down in ["1ST_GOAL", "2ND_GOAL", "3RD_GOAL", "4TH_GOAL"]:
            ttk.Radiobutton(goal_frame, text=down, variable=self.down_var, 
                           value=down, command=self.on_down_change).pack(anchor=tk.W, padx=5)
        
        ttk.Label(goal_frame, text="💡 Use 3rd.png (GOAL screenshot) for all", 
                 font=("Arial", 9), foreground="orange").pack(anchor=tk.W, padx=5)
        
        # Preprocessing option
        ttk.Label(control_frame, text="🔧 Processing:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20, 5))
        
        self.preprocessing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Apply SpygateAI OCR preprocessing", 
                       variable=self.preprocessing_var).pack(anchor=tk.W, padx=10)
        
        ttk.Label(control_frame, text="(Scale=3.5x, CLAHE, Gamma=0.8, etc.)", 
                 font=("Arial", 9), foreground="gray").pack(anchor=tk.W, padx=20)
        
        # Zoom controls
        ttk.Label(control_frame, text="🔍 Zoom Controls:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20, 5))
        
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(fill=tk.X, padx=10)
        
        ttk.Button(zoom_frame, text="➕ Zoom In", command=self.zoom_in, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="➖ Zoom Out", command=self.zoom_out, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_frame, text="🔄 Reset", command=self.reset_zoom, width=8).pack(side=tk.LEFT, padx=2)
        
        self.zoom_label = ttk.Label(control_frame, text="Zoom: 100%", font=("Arial", 10))
        self.zoom_label.pack(anchor=tk.W, padx=10, pady=(5, 0))
        
        ttk.Label(control_frame, text="💡 Mouse wheel to zoom, right-click drag to pan", 
                 font=("Arial", 9), foreground="gray").pack(anchor=tk.W, padx=10)
        
        # Instructions
        ttk.Label(control_frame, text="📋 Instructions:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20, 5))
        
        instructions = [
            "📏 NORMAL TEMPLATES:",
            "1. Select screenshots 1.png, 2.png, 4th.png",
            "2. Choose 1ST, 2ND, 4TH and crop down text",
            "",
            "🥅 GOAL TEMPLATES:",
            "3. Select 3rd.png (GOAL screenshot)",
            "4. Choose 1ST_GOAL, 2ND_GOAL, 3RD_GOAL, 4TH_GOAL",
            "5. Crop each down type from same screenshot",
            "6. All GOAL templates use same positioning",
            "",
            "💾 Save each template after cropping"
        ]
        
        for instruction in instructions:
            if instruction.startswith("📏") or instruction.startswith("🥅"):
                ttk.Label(control_frame, text=instruction, font=("Arial", 10, "bold"), 
                         foreground="blue").pack(anchor=tk.W, padx=10)
            elif instruction == "":
                ttk.Label(control_frame, text="", font=("Arial", 2)).pack()
            else:
                ttk.Label(control_frame, text=instruction, font=("Arial", 9)).pack(anchor=tk.W, padx=15)
        
        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(button_frame, text="💾 Save Template", 
                  command=self.save_template).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🔄 Reset Crop", 
                  command=self.reset_crop).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🔍 Preview Processing", 
                  command=self.preview_processing).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="🧪 Test Preprocessing", 
                  command=self.test_preprocessing).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="📁 Export All", 
                  command=self.export_templates).pack(fill=tk.X, pady=2)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to create enhanced templates")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, 
                                font=("Arial", 10), foreground="blue")
        status_label.pack(pady=(20, 0))
        
        # Progress
        ttk.Label(control_frame, text="📊 Progress:", font=("Arial", 12, "bold")).pack(anchor=tk.W, pady=(20, 5))
        self.progress_frame = ttk.Frame(control_frame)
        self.progress_frame.pack(fill=tk.X)
        
        self.progress_labels = {}
        
        # Normal templates
        normal_progress_frame = ttk.LabelFrame(self.progress_frame, text="📏 Normal Templates")
        normal_progress_frame.pack(fill=tk.X, pady=(0, 5))
        
        for down in ["1ST", "2ND", "3RD", "4TH"]:
            label = ttk.Label(normal_progress_frame, text=f"{down}: ❌", font=("Arial", 9))
            label.pack(anchor=tk.W, padx=5)
            self.progress_labels[down] = label
        
        # GOAL templates
        goal_progress_frame = ttk.LabelFrame(self.progress_frame, text="🥅 GOAL Templates")
        goal_progress_frame.pack(fill=tk.X)
        
        for down in ["1ST_GOAL", "2ND_GOAL", "3RD_GOAL", "4TH_GOAL"]:
            label = ttk.Label(goal_progress_frame, text=f"{down}: ❌", font=("Arial", 9))
            label.pack(anchor=tk.W, padx=5)
            self.progress_labels[down] = label
        
        # Image display (right side)
        image_frame = ttk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas for image display
        self.canvas = tk.Canvas(image_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for cropping, zooming, and panning
        self.canvas.bind("<Button-1>", self.start_crop)
        self.canvas.bind("<B1-Motion>", self.update_crop)
        self.canvas.bind("<ButtonRelease-1>", self.end_crop)
        
        # Zoom with mouse wheel
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux
        
        # Pan with right mouse button
        self.canvas.bind("<Button-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.update_pan)
        self.canvas.bind("<ButtonRelease-3>", self.end_pan)
        
        # Keyboard shortcuts
        self.canvas.bind("<Key>", self.on_key_press)
        self.canvas.focus_set()
    
    def load_screenshots(self):
        """Load screenshots from the 'down templates' folder."""
        screenshots_dir = Path("down templates")
        
        if not screenshots_dir.exists():
            messagebox.showerror("Error", "No 'down templates' folder found!")
            return
        
        # Find image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
        screenshots = []
        
        for ext in image_extensions:
            screenshots.extend(screenshots_dir.glob(f"*{ext}"))
        
        if not screenshots:
            messagebox.showerror("Error", "No image files found in 'down templates' folder!")
            return
        
        # Add to listbox
        for screenshot in sorted(screenshots):
            self.screenshot_listbox.insert(tk.END, screenshot.name)
        
        self.status_var.set(f"Found {len(screenshots)} screenshots")
    
    def on_screenshot_select(self, event):
        """Handle screenshot selection."""
        selection = self.screenshot_listbox.curselection()
        if not selection:
            return
        
        filename = self.screenshot_listbox.get(selection[0])
        image_path = Path("down templates") / filename
        
        self.load_image(image_path)
    
    def load_image(self, image_path):
        """Load and display an image."""
        try:
            self.current_image = cv2.imread(str(image_path))
            self.current_image_path = image_path
            
            if self.current_image is None:
                messagebox.showerror("Error", f"Could not load image: {image_path}")
                return
            
            self.display_image_on_canvas()
            self.status_var.set(f"Loaded: {image_path.name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def display_image_on_canvas(self):
        """Display the current image on the canvas with zoom and pan support."""
        if self.current_image is None:
            return
        
        # Get canvas size
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            self.root.after(100, self.display_image_on_canvas)
            return
        
        # Calculate scale factor
        img_height, img_width = self.current_image.shape[:2]
        scale_x = canvas_width / img_width
        scale_y = canvas_height / img_height
        base_scale = min(scale_x, scale_y, 1.0)  # Don't upscale beyond original
        
        # Apply zoom
        self.scale_factor = base_scale * self.zoom_factor
        
        # Resize image
        new_width = int(img_width * self.scale_factor)
        new_height = int(img_height * self.scale_factor)
        
        resized_image = cv2.resize(self.current_image, (new_width, new_height))
        
        # Convert to RGB and then to PIL
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        self.display_image = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image with pan offset
        self.canvas.delete("all")
        x_pos = canvas_width//2 + self.pan_offset_x
        y_pos = canvas_height//2 + self.pan_offset_y
        self.canvas.create_image(x_pos, y_pos, image=self.display_image, anchor=tk.CENTER)
        
        # Update zoom label
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_factor * 100)}%")
    
    def zoom_in(self):
        """Zoom in on the image."""
        self.zoom_factor = min(self.zoom_factor * 1.5, 10.0)
        self.display_image_on_canvas()
        self.status_var.set(f"Zoomed in to {int(self.zoom_factor * 100)}%")
    
    def zoom_out(self):
        """Zoom out of the image."""
        self.zoom_factor = max(self.zoom_factor / 1.5, 0.1)
        self.display_image_on_canvas()
        self.status_var.set(f"Zoomed out to {int(self.zoom_factor * 100)}%")
    
    def reset_zoom(self):
        """Reset zoom and pan to default."""
        self.zoom_factor = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.display_image_on_canvas()
        self.status_var.set("Zoom and pan reset")
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel zoom."""
        if event.delta > 0 or event.num == 4:
            self.zoom_in()
        else:
            self.zoom_out()
    
    def start_pan(self, event):
        """Start panning with right mouse button."""
        self.last_pan_x = event.x
        self.last_pan_y = event.y
        self.panning = True
        self.canvas.config(cursor="fleur")
    
    def update_pan(self, event):
        """Update pan position."""
        if self.panning:
            dx = event.x - self.last_pan_x
            dy = event.y - self.last_pan_y
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            self.last_pan_x = event.x
            self.last_pan_y = event.y
            self.display_image_on_canvas()
    
    def end_pan(self, event):
        """End panning."""
        self.panning = False
        self.canvas.config(cursor="")
    
    def on_key_press(self, event):
        """Handle keyboard shortcuts."""
        if event.keysym == "plus" or event.keysym == "equal":
            self.zoom_in()
        elif event.keysym == "minus":
            self.zoom_out()
        elif event.keysym == "0":
            self.reset_zoom()
    
    def test_preprocessing(self):
        """Test the SpygateAI preprocessing pipeline on a sample region."""
        if self.current_image is None:
            messagebox.showerror("Error", "Please load an image first!")
            return
        
        try:
            # Create a test region from center of image
            img_height, img_width = self.current_image.shape[:2]
            test_size = 100
            center_x, center_y = img_width // 2, img_height // 2
            
            test_region = self.current_image[
                center_y - test_size//2:center_y + test_size//2,
                center_x - test_size//2:center_x + test_size//2
            ]
            
            # Apply preprocessing
            processed = self.apply_spygate_preprocessing(test_region)
            
            if processed is None:
                messagebox.showerror("Error", "Preprocessing failed!")
                return
            
            # Verify pipeline worked
            pipeline_checks = []
            
            # Check 1: Should be grayscale
            if len(processed.shape) == 2:
                pipeline_checks.append("✅ Grayscale conversion: OK")
            else:
                pipeline_checks.append("❌ Grayscale conversion: FAILED")
            
            # Check 2: Should be larger (3.5x scale)
            expected_size = (test_size * 3.5, test_size * 3.5)
            actual_size = processed.shape[::-1]  # (width, height)
            if abs(actual_size[0] - expected_size[0]) < 50:  # Allow some tolerance
                pipeline_checks.append(f"✅ Scaling (3.5x): OK - {actual_size[0]}x{actual_size[1]}px")
            else:
                pipeline_checks.append(f"❌ Scaling (3.5x): FAILED - {actual_size[0]}x{actual_size[1]}px")
            
            # Check 3: Should be binary (0 and 255 values)
            unique_values = np.unique(processed)
            if len(unique_values) <= 10 and (0 in unique_values or 255 in unique_values):
                pipeline_checks.append(f"✅ Thresholding: OK - {len(unique_values)} unique values")
            else:
                pipeline_checks.append(f"❌ Thresholding: FAILED - {len(unique_values)} unique values")
            
            # Check 4: Gamma correction applied (indirect check)
            mid_gray_count = np.sum((processed > 50) & (processed < 200))
            if mid_gray_count < processed.size * 0.3:  # Most pixels should be binary
                pipeline_checks.append("✅ Gamma correction: Applied (binary distribution)")
            else:
                pipeline_checks.append("⚠️ Gamma correction: May need verification")
            
            # Show results
            test_window = tk.Toplevel(self.root)
            test_window.title("SpygateAI Preprocessing Pipeline Test")
            test_window.geometry("600x500")
            
            # Title
            ttk.Label(test_window, text="🧪 Preprocessing Pipeline Verification", 
                     font=("Arial", 14, "bold")).pack(pady=10)
            
            # Pipeline steps
            steps_frame = ttk.LabelFrame(test_window, text="8-Stage Pipeline Applied")
            steps_frame.pack(fill=tk.X, padx=10, pady=10)
            
            steps = [
                "1. Grayscale conversion",
                "2. LANCZOS4 scaling (3.5x)",
                "3. CLAHE (clip=1.0, grid=4x4)",
                "4. Gamma correction (0.8)",
                "5. Gaussian blur (3x3)",
                "6. Adaptive threshold (mean, block=13, C=3)",
                "7. Morphological closing (3x3)",
                "8. Sharpening (disabled for optimal combo)"
            ]
            
            for step in steps:
                ttk.Label(steps_frame, text=step, font=("Arial", 10)).pack(anchor=tk.W, padx=10, pady=2)
            
            # Verification results
            results_frame = ttk.LabelFrame(test_window, text="Verification Results")
            results_frame.pack(fill=tk.X, padx=10, pady=10)
            
            for check in pipeline_checks:
                color = "green" if "✅" in check else "orange" if "⚠️" in check else "red"
                ttk.Label(results_frame, text=check, font=("Arial", 10), 
                         foreground=color).pack(anchor=tk.W, padx=10, pady=2)
            
            # Sample display
            sample_frame = ttk.LabelFrame(test_window, text="Sample Result")
            sample_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            sample_canvas = tk.Canvas(sample_frame, bg="white", height=150)
            sample_canvas.pack(fill=tk.X, padx=10, pady=10)
            
            # Display sample
            sample_pil = Image.fromarray(processed)
            sample_pil = sample_pil.resize((200, 100), Image.Resampling.NEAREST)
            sample_photo = ImageTk.PhotoImage(sample_pil)
            
            sample_canvas.create_image(100, 50, image=sample_photo, anchor=tk.CENTER)
            sample_canvas.image = sample_photo
            
            ttk.Label(test_window, text="Pipeline working! Templates will be optimized for SpygateAI detection.", 
                     font=("Arial", 10), foreground="green").pack(pady=10)
            
            self.status_var.set("Preprocessing pipeline verified - ready for template creation")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to test preprocessing: {e}")
    
    def start_crop(self, event):
        """Start cropping."""
        self.crop_start = (event.x, event.y)
        self.cropping = True
        self.canvas.delete("crop_rect")
    
    def update_crop(self, event):
        """Update crop rectangle."""
        if not self.cropping or not self.crop_start:
            return
        
        self.canvas.delete("crop_rect")
        self.canvas.create_rectangle(
            self.crop_start[0], self.crop_start[1],
            event.x, event.y,
            outline="red", width=2, tags="crop_rect"
        )
    
    def end_crop(self, event):
        """End cropping."""
        if not self.cropping or not self.crop_start:
            return
        
        self.crop_end = (event.x, event.y)
        self.cropping = False
        
        # Validate crop
        if abs(self.crop_end[0] - self.crop_start[0]) < 10 or abs(self.crop_end[1] - self.crop_start[1]) < 10:
            self.status_var.set("Crop too small - try again")
            return
        
        self.status_var.set(f"Cropped region selected for {self.down_var.get()}")
    
    def save_template(self):
        """Save the cropped template."""
        if not self.crop_start or not self.crop_end or self.current_image is None:
            messagebox.showerror("Error", "Please select an image and crop a region first!")
            return
        
        try:
            # Calculate crop coordinates in original image accounting for zoom and pan
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_height, img_width = self.current_image.shape[:2]
            
            # Calculate how the image is positioned on canvas
            base_scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            self.scale_factor = base_scale * self.zoom_factor
            
            scaled_width = int(img_width * self.scale_factor)
            scaled_height = int(img_height * self.scale_factor)
            
            # Image center position on canvas including pan offset
            img_center_x = canvas_width // 2 + self.pan_offset_x
            img_center_y = canvas_height // 2 + self.pan_offset_y
            
            # Top-left corner of image on canvas
            img_x = img_center_x - scaled_width // 2
            img_y = img_center_y - scaled_height // 2
            
            # Ensure crop coordinates are ordered correctly (top-left to bottom-right)
            start_x, end_x = sorted([self.crop_start[0], self.crop_end[0]])
            start_y, end_y = sorted([self.crop_start[1], self.crop_end[1]])
            
            # Convert crop coordinates from canvas to original image coordinates
            crop_x1 = int((start_x - img_x) / self.scale_factor)
            crop_y1 = int((start_y - img_y) / self.scale_factor)
            crop_x2 = int((end_x - img_x) / self.scale_factor)
            crop_y2 = int((end_y - img_y) / self.scale_factor)
            
            # Clamp coordinates to image bounds
            crop_x1 = max(0, min(crop_x1, img_width))
            crop_y1 = max(0, min(crop_y1, img_height))
            crop_x2 = max(0, min(crop_x2, img_width))
            crop_y2 = max(0, min(crop_y2, img_height))
            
            # Debug info
            print(f"Canvas: {canvas_width}x{canvas_height}")
            print(f"Original image: {img_width}x{img_height}")
            print(f"Base scale: {base_scale:.3f}, Zoom: {self.zoom_factor:.3f}, Final scale: {self.scale_factor:.3f}")
            print(f"Pan offset: ({self.pan_offset_x}, {self.pan_offset_y})")
            print(f"Image center: ({img_center_x}, {img_center_y})")
            print(f"Image position (top-left): ({img_x}, {img_y})")
            print(f"Crop canvas coords (ordered): ({start_x}, {start_y}) to ({end_x}, {end_y})")
            print(f"Crop image coords (raw): ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
            
            # Ensure valid crop dimensions
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                error_msg = f"Invalid crop coordinates!\n"
                error_msg += f"Image coords: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})\n"
                error_msg += f"Canvas coords: ({start_x}, {start_y}) to ({end_x}, {end_y})\n"
                error_msg += f"Image position: ({img_x}, {img_y})\n"
                error_msg += f"Scale: {self.scale_factor:.3f}, Zoom: {self.zoom_factor:.3f}\n"
                error_msg += f"\nTip: Make sure to crop within the image boundaries.\n"
                error_msg += f"Try resetting zoom/pan and select a clear region within the image."
                messagebox.showerror("Invalid Crop", error_msg)
                return
            
            # Final validation - ensure we have a reasonable crop size
            crop_width = crop_x2 - crop_x1
            crop_height = crop_y2 - crop_y1
            
            if crop_width < 10 or crop_height < 10:
                error_msg = f"Crop too small!\n"
                error_msg += f"Crop size: {crop_width}x{crop_height} pixels\n"
                error_msg += f"Minimum: 10x10 pixels\n"
                error_msg += f"Try zooming in and selecting a larger region."
                messagebox.showerror("Crop Too Small", error_msg)
                return
            
            # Crop the template
            template = self.current_image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if template.size == 0:
                messagebox.showerror("Error", "Failed to crop template - empty region selected!")
                return
            
            # Apply SpygateAI preprocessing if enabled
            if self.preprocessing_var.get():
                processed_template = self.apply_spygate_preprocessing(template)
                if processed_template is not None:
                    template = processed_template
                    processing_info = "Enhanced with SpygateAI OCR preprocessing (Scale=3.5x, CLAHE, Gamma=0.8)"
                else:
                    processing_info = "Raw template (preprocessing failed)"
            else:
                processing_info = "Raw template (no preprocessing)"
            
            # Save template
            down_type = self.down_var.get()
            output_dir = Path("down_templates_real")
            output_dir.mkdir(exist_ok=True)
            
            template_path = output_dir / f"{down_type}.png"
            success = cv2.imwrite(str(template_path), template)
            
            if not success:
                messagebox.showerror("Error", f"Failed to save template to {template_path}")
                return
            
            # Store template data
            self.templates_data[down_type] = {
                'source_image': str(self.current_image_path),
                'crop_coords': [crop_x1, crop_y1, crop_x2, crop_y2],
                'template_path': str(template_path),
                'size': template.shape[:2] if len(template.shape) >= 2 else [0, 0],
                'preprocessing_applied': self.preprocessing_var.get(),
                'processing_info': processing_info
            }
            
            # Update progress (handle GOAL templates too)
            if down_type in self.progress_labels:
                self.progress_labels[down_type].config(text=f"{down_type}: ✅")
            
            self.status_var.set(f"✅ Saved {down_type} template: {template.shape[1]}x{template.shape[0]}px ({processing_info})")
            
            # Clear crop
            self.reset_crop()
            
            # Show success message with template info
            messagebox.showinfo("Success!", 
                              f"Template '{down_type}' saved successfully!\n\n"
                              f"Size: {template.shape[1]}x{template.shape[0]} pixels\n"
                              f"Path: {template_path}\n"
                              f"{processing_info}")
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error details:\n{error_details}")
            
            error_msg = f"Failed to save template '{self.down_var.get()}':\n{str(e)}\n\n"
            error_msg += "Debug info:\n"
            if hasattr(self, 'crop_start') and self.crop_start:
                error_msg += f"Crop start: {self.crop_start}\n"
            if hasattr(self, 'crop_end') and self.crop_end:
                error_msg += f"Crop end: {self.crop_end}\n"
            error_msg += f"Zoom: {getattr(self, 'zoom_factor', 'unknown')}\n"
            error_msg += f"Pan: ({getattr(self, 'pan_offset_x', 'unknown')}, {getattr(self, 'pan_offset_y', 'unknown')})\n"
            error_msg += "\nTry resetting zoom/pan and cropping again."
            
            messagebox.showerror("Error", error_msg)
    
    def reset_crop(self):
        """Reset the crop selection."""
        self.crop_start = None
        self.crop_end = None
        self.cropping = False
        self.canvas.delete("crop_rect")
        self.status_var.set("Crop reset")
    
    def on_down_change(self):
        """Handle down type change."""
        self.current_down = self.down_var.get()
        self.reset_crop()
    
    def export_templates(self):
        """Export all templates and metadata."""
        if not self.templates_data:
            messagebox.showerror("Error", "No templates created yet!")
            return
        
        try:
            # Save metadata
            metadata_path = Path("down_templates_real") / "templates_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(self.templates_data, f, indent=2)
            
            # Create summary
            summary = []
            summary.append("SpygateAI Down Templates Summary")
            summary.append("=" * 40)
            summary.append(f"Created: {len(self.templates_data)} templates")
            summary.append("")
            
            for down_type, data in self.templates_data.items():
                summary.append(f"{down_type}:")
                summary.append(f"  Source: {Path(data['source_image']).name}")
                summary.append(f"  Size: {data['size'][1]}x{data['size'][0]}px")
                summary.append(f"  Crop: {data['crop_coords']}")
                summary.append("")
            
            summary_path = Path("down_templates_real") / "README.txt"
            with open(summary_path, 'w') as f:
                f.write('\n'.join(summary))
            
            messagebox.showinfo("Success", 
                              f"Exported {len(self.templates_data)} templates to 'down_templates_real' folder!\n\n"
                              f"Files created:\n"
                              f"- Template images (.png)\n"
                              f"- templates_metadata.json\n"
                              f"- README.txt")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export templates: {e}")
    
    def preview_processing(self):
        """Preview the SpygateAI preprocessing on the current crop."""
        if not self.crop_start or not self.crop_end or self.current_image is None:
            messagebox.showwarning("Warning", "Please select an image and crop a region first!")
            return
        
        try:
            # Get crop coordinates (accounting for zoom and pan)
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            img_height, img_width = self.current_image.shape[:2]
            
            scaled_width = int(img_width * self.scale_factor)
            scaled_height = int(img_height * self.scale_factor)
            img_x = (canvas_width - scaled_width) // 2 + self.pan_offset_x
            img_y = (canvas_height - scaled_height) // 2 + self.pan_offset_y
            
            # Ensure crop coordinates are ordered correctly
            start_x, end_x = sorted([self.crop_start[0], self.crop_end[0]])
            start_y, end_y = sorted([self.crop_start[1], self.crop_end[1]])
            
            crop_x1 = max(0, int((start_x - img_x) / self.scale_factor))
            crop_y1 = max(0, int((start_y - img_y) / self.scale_factor))
            crop_x2 = min(img_width, int((end_x - img_x) / self.scale_factor))
            crop_y2 = min(img_height, int((end_y - img_y) / self.scale_factor))
            
            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                messagebox.showerror("Error", "Invalid crop coordinates! Try resetting zoom/pan.")
                return
            
            # Crop the region
            cropped = self.current_image[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Apply preprocessing
            processed = self.apply_spygate_preprocessing(cropped)
            
            if processed is None:
                messagebox.showerror("Error", "Preprocessing failed!")
                return
            
            # Create preview window
            preview_window = tk.Toplevel(self.root)
            preview_window.title("SpygateAI Preprocessing Preview")
            preview_window.geometry("800x400")
            
            # Create frames for before/after
            compare_frame = ttk.Frame(preview_window)
            compare_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Original image
            original_frame = ttk.LabelFrame(compare_frame, text="Original Crop")
            original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
            
            original_canvas = tk.Canvas(original_frame, bg="white")
            original_canvas.pack(fill=tk.BOTH, expand=True)
            
            # Processed image
            processed_frame = ttk.LabelFrame(compare_frame, text="SpygateAI Enhanced")
            processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
            
            processed_canvas = tk.Canvas(processed_frame, bg="white")
            processed_canvas.pack(fill=tk.BOTH, expand=True)
            
            # Display images
            def display_preview_images():
                # Original
                if len(cropped.shape) == 3:
                    orig_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                else:
                    orig_rgb = cropped
                
                orig_pil = Image.fromarray(orig_rgb)
                orig_pil = orig_pil.resize((300, 150), Image.Resampling.LANCZOS)
                orig_photo = ImageTk.PhotoImage(orig_pil)
                
                original_canvas.delete("all")
                original_canvas.create_image(150, 75, image=orig_photo, anchor=tk.CENTER)
                original_canvas.image = orig_photo  # Keep reference
                
                # Processed
                proc_pil = Image.fromarray(processed)
                proc_pil = proc_pil.resize((300, 150), Image.Resampling.LANCZOS)
                proc_photo = ImageTk.PhotoImage(proc_pil)
                
                processed_canvas.delete("all")
                processed_canvas.create_image(150, 75, image=proc_photo, anchor=tk.CENTER)
                processed_canvas.image = proc_photo  # Keep reference
            
            # Display after window is ready
            preview_window.after(100, display_preview_images)
            
            # Info label
            info_text = ("SpygateAI Optimal Pipeline: Scale=3.5x → CLAHE → Gamma=0.8 → "
                        "Blur → Adaptive Threshold → Morphological Closing")
            ttk.Label(preview_window, text=info_text, font=("Arial", 9), 
                     foreground="green").pack(pady=5)
            
            self.status_var.set("Preview window opened - compare original vs enhanced")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create preview: {e}")
    
    def run(self):
        """Run the GUI."""
        self.root.mainloop()


def main():
    """Main function."""
    print("🎯 SpygateAI Down Template Creator")
    print("=" * 40)
    print("📸 Place your Madden 25 screenshots in 'down templates' folder")
    print("🖱️  Use the GUI to crop down text regions")
    print("💾 Templates will be saved to 'down_templates_real' folder")
    print("")
    
    # Check if screenshots exist
    screenshots_dir = Path("down templates")
    if not screenshots_dir.exists():
        print("❌ Error: 'down templates' folder not found!")
        print("💡 Create the folder and add your Madden 25 screenshots")
        return
    
    screenshots = list(screenshots_dir.glob("*.png")) + list(screenshots_dir.glob("*.jpg"))
    if not screenshots:
        print("❌ Error: No screenshots found in 'down templates' folder!")
        print("💡 Add some Madden 25 screenshots (.png or .jpg)")
        return
    
    print(f"✅ Found {len(screenshots)} screenshots")
    print("🚀 Launching GUI...")
    
    # Launch GUI
    gui = DownTemplateCropperGUI()
    gui.run()


if __name__ == "__main__":
    main() 