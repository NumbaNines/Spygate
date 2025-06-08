#!/usr/bin/env python3
"""
SpygateAI Model Training Module
This module handles YOLO model training using the ultralytics library.
"""

import os
import logging
from typing import Callable
from ultralytics import YOLO
import yaml

class TrainingCallback:
    """Custom callback for YOLO training."""
    def __init__(self, progress_callback: Callable[[int], None]):
        self.progress_callback = progress_callback

    def __call__(self, trainer):
        """Called by YOLO trainer after each epoch."""
        try:
            # Update progress based on current epoch
            current_epoch = trainer.epoch
            self.progress_callback(current_epoch)
            
            # Log metrics if available
            if hasattr(trainer, 'metrics'):
                metrics = trainer.metrics
                logging.info(
                    f"Epoch {current_epoch}: "
                    f"mAP@.5 = {metrics.get('metrics/mAP50(B)', 0):.3f}, "
                    f"mAP@.5:.95 = {metrics.get('metrics/mAP50-95(B)', 0):.3f}"
                )
        except Exception as e:
            logging.warning(f"Error in progress callback: {str(e)}")

class ModelTrainer:
    """Handles YOLO model training for SpygateAI."""
    
    def __init__(self, data_yaml: str, progress_callback: Callable[[int], None]):
        """
        Initialize the model trainer.
        
        Args:
            data_yaml: Path to the data.yaml configuration file
            progress_callback: Callback function to update training progress
        """
        # Convert to absolute path and normalize
        self.data_yaml = os.path.abspath(data_yaml).replace('\\', '/')
        self.progress_callback = progress_callback
        
        # Set working directory to the dataset root
        self.dataset_dir = os.path.dirname(self.data_yaml)
        os.chdir(self.dataset_dir)
        
        # Verify data.yaml exists
        if not os.path.exists(self.data_yaml):
            raise FileNotFoundError(
                f"Data configuration file not found: {self.data_yaml}"
            )
        
        # Validate dataset structure
        self._validate_dataset()

    def _validate_dataset(self) -> None:
        """Validate the dataset structure and paths."""
        try:
            # Read data.yaml
            with open(self.data_yaml, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Verify required paths exist
            required_dirs = ['train', 'val', 'test']
            for dir_name in required_dirs:
                img_dir = os.path.join(self.dataset_dir, 'images', dir_name)
                if not os.path.exists(img_dir):
                    raise FileNotFoundError(
                        f"Required directory not found: {img_dir}"
                    )
                
                # Check if directory contains images
                if not any(f.lower().endswith(('.png', '.jpg', '.jpeg')) 
                          for f in os.listdir(img_dir)):
                    logging.warning(f"No images found in {img_dir}")
                
                # Check if labels directory exists and contains files
                label_dir = os.path.join(self.dataset_dir, 'labels', dir_name)
                if not os.path.exists(label_dir):
                    raise FileNotFoundError(
                        f"Required directory not found: {label_dir}"
                    )
                
                if not any(f.endswith('.txt') for f in os.listdir(label_dir)):
                    logging.warning(f"No label files found in {label_dir}")
            
            logging.info("Dataset validation completed successfully")
            
        except Exception as e:
            logging.error(f"Dataset validation failed: {str(e)}")
            raise

    def train(self) -> None:
        """Train the YOLO model."""
        try:
            # Load a pre-trained YOLO model (using nano size for faster training)
            model = YOLO('yolov8n.pt')
            
            # Configure training parameters
            results = model.train(
                task='detect',  # Detection task
                data=self.data_yaml,  # Path to data.yaml
                epochs=100,  # Increased epochs for better learning
                imgsz=640,  # Image size
                batch=16,  # Increased batch size
                project=os.path.join(self.dataset_dir, 'runs/train'),  # Project directory
                name='exp',  # Experiment name
                exist_ok=True,  # Overwrite existing experiment
                plots=True,  # Generate training plots
                save=True,  # Save trained model
                device='cpu',  # Force CPU training
                workers=0,  # Number of worker threads (0 for CPU)
                patience=20,  # Early stopping patience
                save_period=10,  # Save checkpoint every 10 epochs
                verbose=True,  # Verbose output
                lr0=0.01,  # Initial learning rate
                lrf=0.001,  # Final learning rate
                momentum=0.937,  # SGD momentum/Adam beta1
                weight_decay=0.0005,  # Optimizer weight decay
                warmup_epochs=3,  # Warmup epochs
                warmup_momentum=0.8,  # Warmup initial momentum
                warmup_bias_lr=0.1,  # Warmup initial bias lr
                box=7.5,  # Box loss gain
                cls=0.5,  # Class loss gain
                dfl=1.5,  # DFL loss gain
                close_mosaic=10,  # Disable mosaic augmentation for final epochs
                amp=False,  # Disable mixed precision (more stable on CPU)
                cache=True  # Cache images for faster training
            )
            
            # Update progress after each epoch
            if hasattr(results, 'epoch'):
                self.progress_callback(results.epoch)
            
            # Validate the model on test set
            metrics = model.val(
                data=self.data_yaml,
                split='test'  # Use test split for validation
            )
            
            logging.info(f"Training completed. Results saved to: {results.save_dir}")
            logging.info(f"Validation metrics: {metrics}")
            
        except Exception as e:
            logging.error(f"An error occurred during model training: {str(e)}")
            raise

    def validate(self, weights_path: str) -> dict:
        """
        Validate a trained model.
        
        Args:
            weights_path: Path to the trained model weights
            
        Returns:
            dict: Validation metrics
        """
        try:
            model = YOLO(weights_path)
            metrics = model.val(data=self.data_yaml)
            return metrics
        except Exception as e:
            logging.error(f"Validation failed: {str(e)}")
            raise