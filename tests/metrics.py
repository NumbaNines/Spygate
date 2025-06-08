"""Test evaluation metrics."""

import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import json

from tests.utils import BoundingBox, calculate_iou

class TrackingMetrics:
    """Calculate metrics for tracking performance evaluation."""
    
    @staticmethod
    def calculate_tracking_accuracy(
        predicted_boxes: List[BoundingBox],
        ground_truth_boxes: List[BoundingBox],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate tracking accuracy metrics.
        
        Args:
            predicted_boxes: List of predicted bounding boxes
            ground_truth_boxes: List of ground truth bounding boxes
            iou_threshold: IoU threshold for successful detection
            
        Returns:
            Dictionary with accuracy metrics
        """
        if len(predicted_boxes) != len(ground_truth_boxes):
            raise ValueError("Number of predictions must match ground truth")
            
        ious = [calculate_iou(pred, gt) for pred, gt in zip(predicted_boxes, ground_truth_boxes)]
        successful_tracks = sum(1 for iou in ious if iou >= iou_threshold)
        
        metrics = {
            'accuracy': successful_tracks / len(predicted_boxes),
            'mean_iou': np.mean(ious),
            'std_iou': np.std(ious),
            'min_iou': np.min(ious),
            'max_iou': np.max(ious)
        }
        
        return metrics
    
    @staticmethod
    def calculate_position_error(
        predicted_positions: List[Tuple[float, float]],
        ground_truth_positions: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """Calculate position error metrics.
        
        Args:
            predicted_positions: List of predicted (x, y) positions
            ground_truth_positions: List of ground truth (x, y) positions
            
        Returns:
            Dictionary with error metrics
        """
        if len(predicted_positions) != len(ground_truth_positions):
            raise ValueError("Number of predictions must match ground truth")
            
        errors = [np.linalg.norm(np.array(pred) - np.array(gt))
                 for pred, gt in zip(predicted_positions, ground_truth_positions)]
        
        metrics = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'min_error': np.min(errors),
            'max_error': np.max(errors),
            'rmse': np.sqrt(np.mean(np.array(errors) ** 2))
        }
        
        return metrics
    
    @staticmethod
    def calculate_velocity_error(
        predicted_velocities: List[Tuple[float, float]],
        ground_truth_velocities: List[Tuple[float, float]]
    ) -> Dict[str, float]:
        """Calculate velocity error metrics.
        
        Args:
            predicted_velocities: List of predicted (vx, vy) velocities
            ground_truth_velocities: List of ground truth velocities
            
        Returns:
            Dictionary with error metrics
        """
        if len(predicted_velocities) != len(ground_truth_velocities):
            raise ValueError("Number of predictions must match ground truth")
            
        magnitude_errors = []
        direction_errors = []
        
        for pred, gt in zip(predicted_velocities, ground_truth_velocities):
            pred_vec = np.array(pred)
            gt_vec = np.array(gt)
            
            # Magnitude error
            pred_mag = np.linalg.norm(pred_vec)
            gt_mag = np.linalg.norm(gt_vec)
            magnitude_errors.append(abs(pred_mag - gt_mag))
            
            # Direction error (in radians)
            if gt_mag > 0 and pred_mag > 0:
                cos_theta = np.dot(pred_vec, gt_vec) / (pred_mag * gt_mag)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Handle numerical errors
                direction_errors.append(np.arccos(cos_theta))
        
        metrics = {
            'mean_magnitude_error': np.mean(magnitude_errors),
            'std_magnitude_error': np.std(magnitude_errors),
            'mean_direction_error_rad': np.mean(direction_errors) if direction_errors else 0.0,
            'mean_direction_error_deg': np.degrees(np.mean(direction_errors)) if direction_errors else 0.0
        }
        
        return metrics
    
    @staticmethod
    def calculate_occlusion_metrics(
        predicted_boxes: List[BoundingBox],
        ground_truth_boxes: List[BoundingBox],
        occlusion_frames: List[int],
        iou_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Calculate metrics for occlusion handling.
        
        Args:
            predicted_boxes: List of predicted bounding boxes
            ground_truth_boxes: List of ground truth bounding boxes
            occlusion_frames: List of frame indices with occlusion
            iou_threshold: IoU threshold for successful detection
            
        Returns:
            Dictionary with occlusion metrics
        """
        if len(predicted_boxes) != len(ground_truth_boxes):
            raise ValueError("Number of predictions must match ground truth")
            
        # Calculate IoU for all frames
        ious = [calculate_iou(pred, gt) for pred, gt in zip(predicted_boxes, ground_truth_boxes)]
        
        # Separate occlusion and non-occlusion frames
        occlusion_ious = [ious[i] for i in occlusion_frames]
        non_occlusion_ious = [ious[i] for i in range(len(ious)) if i not in occlusion_frames]
        
        # Calculate success rates
        occlusion_success = sum(1 for iou in occlusion_ious if iou >= iou_threshold)
        non_occlusion_success = sum(1 for iou in non_occlusion_ious if iou >= iou_threshold)
        
        metrics = {
            'occlusion_accuracy': occlusion_success / len(occlusion_frames) if occlusion_frames else 1.0,
            'non_occlusion_accuracy': non_occlusion_success / len(non_occlusion_ious) if non_occlusion_ious else 1.0,
            'mean_occlusion_iou': np.mean(occlusion_ious) if occlusion_ious else 0.0,
            'mean_non_occlusion_iou': np.mean(non_occlusion_ious) if non_occlusion_ious else 0.0,
            'occlusion_recovery_rate': occlusion_success / len(occlusion_frames) if occlusion_frames else 1.0
        }
        
        return metrics
    
    @staticmethod
    def calculate_formation_metrics(
        predicted_positions: List[List[Tuple[float, float]]],
        ground_truth_positions: List[List[Tuple[float, float]]],
        formation_type: str
    ) -> Dict[str, float]:
        """Calculate metrics for formation tracking.
        
        Args:
            predicted_positions: List of lists of predicted player positions
            ground_truth_positions: List of lists of ground truth positions
            formation_type: Type of formation being tracked
            
        Returns:
            Dictionary with formation metrics
        """
        if len(predicted_positions) != len(ground_truth_positions):
            raise ValueError("Number of frames must match")
            
        position_errors = []
        formation_structure_errors = []
        
        for pred_frame, gt_frame in zip(predicted_positions, ground_truth_positions):
            if len(pred_frame) != len(gt_frame):
                raise ValueError("Number of players must match")
                
            # Calculate position errors
            errors = [np.linalg.norm(np.array(pred) - np.array(gt))
                     for pred, gt in zip(pred_frame, gt_frame)]
            position_errors.append(np.mean(errors))
            
            # Calculate formation structure preservation
            pred_centroid = np.mean(pred_frame, axis=0)
            gt_centroid = np.mean(gt_frame, axis=0)
            
            pred_relative = [np.array(pos) - pred_centroid for pos in pred_frame]
            gt_relative = [np.array(pos) - gt_centroid for pos in gt_frame]
            
            structure_error = np.mean([np.linalg.norm(pred - gt)
                                     for pred, gt in zip(pred_relative, gt_relative)])
            formation_structure_errors.append(structure_error)
        
        metrics = {
            'mean_position_error': np.mean(position_errors),
            'std_position_error': np.std(position_errors),
            'mean_structure_error': np.mean(formation_structure_errors),
            'std_structure_error': np.std(formation_structure_errors),
            'formation_stability': 1.0 / (1.0 + np.mean(formation_structure_errors))
        }
        
        return metrics
    
    @staticmethod
    def calculate_performance_metrics(
        processing_times: List[float],
        memory_usage: List[float],
        gpu_usage: Optional[List[float]] = None,
        target_fps: float = 30.0
    ) -> Dict[str, float]:
        """Calculate performance metrics.
        
        Args:
            processing_times: List of frame processing times (seconds)
            memory_usage: List of memory usage values (MB)
            gpu_usage: Optional list of GPU usage percentages
            target_fps: Target frames per second
            
        Returns:
            Dictionary with performance metrics
        """
        fps = 1.0 / np.mean(processing_times)
        
        metrics = {
            'mean_processing_time': np.mean(processing_times),
            'std_processing_time': np.std(processing_times),
            'fps': fps,
            'fps_stability': 1.0 - (np.std(processing_times) / np.mean(processing_times)),
            'target_fps_ratio': fps / target_fps,
            'mean_memory_usage': np.mean(memory_usage),
            'peak_memory_usage': np.max(memory_usage),
            'memory_efficiency': 1.0 - (np.std(memory_usage) / np.mean(memory_usage))
        }
        
        if gpu_usage:
            metrics.update({
                'mean_gpu_usage': np.mean(gpu_usage),
                'peak_gpu_usage': np.max(gpu_usage),
                'gpu_efficiency': 1.0 - (np.std(gpu_usage) / np.mean(gpu_usage))
            })
        
        return metrics
    
    @staticmethod
    def generate_summary_report(
        accuracy_metrics: Dict[str, float],
        position_metrics: Dict[str, float],
        occlusion_metrics: Dict[str, float],
        performance_metrics: Dict[str, float],
        output_file: Optional[Path] = None
    ) -> Dict[str, Dict[str, float]]:
        """Generate a comprehensive summary report.
        
        Args:
            accuracy_metrics: Tracking accuracy metrics
            position_metrics: Position error metrics
            occlusion_metrics: Occlusion handling metrics
            performance_metrics: Performance metrics
            output_file: Optional file to save the report
            
        Returns:
            Dictionary with all metrics
        """
        report = {
            'tracking_accuracy': accuracy_metrics,
            'position_accuracy': position_metrics,
            'occlusion_handling': occlusion_metrics,
            'performance': performance_metrics,
            'overall_scores': {
                'tracking_score': accuracy_metrics['accuracy'],
                'position_score': 1.0 / (1.0 + position_metrics['mean_error']),
                'occlusion_score': occlusion_metrics['occlusion_recovery_rate'],
                'performance_score': performance_metrics['fps_stability']
            }
        }
        
        # Calculate weighted overall score
        weights = {
            'tracking': 0.4,
            'position': 0.3,
            'occlusion': 0.2,
            'performance': 0.1
        }
        
        overall_score = (
            weights['tracking'] * report['overall_scores']['tracking_score'] +
            weights['position'] * report['overall_scores']['position_score'] +
            weights['occlusion'] * report['overall_scores']['occlusion_score'] +
            weights['performance'] * report['overall_scores']['performance_score']
        )
        
        report['overall_scores']['final_score'] = overall_score
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report 