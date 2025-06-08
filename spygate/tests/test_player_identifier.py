"""
Tests for the player identification module.
"""

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch

from ..utils.tracking_hardware import TrackingMode
from ..video.player_identifier import PlayerIdentifier


class TestPlayerIdentifier(unittest.TestCase):
    """Test cases for PlayerIdentifier class."""

    def setUp(self):
        """Set up test fixtures."""
        self.identifier = PlayerIdentifier(confidence_threshold=0.7)

        # Create a sample test image
        self.test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(
            self.test_image,
            "23",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Create a sample face image
        self.face_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(self.face_image, (50, 50), 30, (255, 255, 255), -1)
        cv2.circle(self.face_image, (40, 40), 5, (0, 0, 0), -1)
        cv2.circle(self.face_image, (60, 40), 5, (0, 0, 0), -1)
        cv2.ellipse(self.face_image, (50, 60), (20, 10), 0, 0, 180, (0, 0, 0), 2)

    def test_initialization(self):
        """Test initialization with different hardware modes."""
        # Test with BASIC mode
        with patch("torch.cuda.is_available", return_value=False):
            with patch(
                "spygate.utils.tracking_hardware.TrackingHardwareManager"
            ) as mock_manager:
                mock_manager.return_value.tracking_mode = TrackingMode.BASIC
                identifier = PlayerIdentifier()
                self.assertFalse(identifier.use_face_recognition)

        # Test with PROFESSIONAL mode and GPU
        with patch("torch.cuda.is_available", return_value=True):
            with patch(
                "spygate.utils.tracking_hardware.TrackingHardwareManager"
            ) as mock_manager:
                mock_manager.return_value.tracking_mode = TrackingMode.PROFESSIONAL
                identifier = PlayerIdentifier()
                self.assertTrue(identifier.use_face_recognition)

    @patch("easyocr.Reader")
    def test_jersey_number_recognition(self, mock_reader):
        """Test jersey number recognition."""
        # Mock OCR results
        mock_reader.return_value.readtext.return_value = [(None, "23", 0.95)]

        # Test successful recognition
        result = self.identifier._recognize_jersey_number(self.test_image)
        self.assertIsNotNone(result)
        self.assertEqual(result["number"], "23")
        self.assertGreater(result["confidence"], 0.7)

        # Test low confidence
        mock_reader.return_value.readtext.return_value = [(None, "23", 0.5)]
        result = self.identifier._recognize_jersey_number(self.test_image)
        self.assertIsNone(result)

        # Test invalid number
        mock_reader.return_value.readtext.return_value = [(None, "ABC", 0.95)]
        result = self.identifier._recognize_jersey_number(self.test_image)
        self.assertIsNone(result)

    @patch("face_recognition.face_locations")
    @patch("face_recognition.face_encodings")
    @patch("face_recognition.face_distance")
    def test_face_recognition(self, mock_distance, mock_encodings, mock_locations):
        """Test face recognition."""
        # Enable face recognition
        self.identifier.use_face_recognition = True

        # Mock face detection
        mock_locations.return_value = [(10, 10, 50, 50)]
        mock_encodings.return_value = [np.array([0.1, 0.2, 0.3])]

        # Test successful match
        mock_distance.return_value = np.array([0.3])
        reference_faces = {"player1": np.array([0.15, 0.25, 0.35])}

        result = self.identifier._match_face(self.face_image, reference_faces)
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "player1")
        self.assertGreater(result["confidence"], 0.7)

        # Test no match (high distance)
        mock_distance.return_value = np.array([0.8])
        result = self.identifier._match_face(self.face_image, reference_faces)
        self.assertIsNone(result)

        # Test no face detected
        mock_locations.return_value = []
        result = self.identifier._match_face(self.face_image, reference_faces)
        self.assertIsNone(result)

    def test_add_reference_face(self):
        """Test adding reference faces."""
        # Enable face recognition
        self.identifier.use_face_recognition = True

        with patch("face_recognition.face_locations") as mock_locations:
            with patch("face_recognition.face_encodings") as mock_encodings:
                # Test successful addition
                mock_locations.return_value = [(10, 10, 50, 50)]
                mock_encodings.return_value = [np.array([0.1, 0.2, 0.3])]

                success = self.identifier.add_reference_face("player1", self.face_image)
                self.assertTrue(success)
                self.assertIn("player1", self.identifier.face_encodings_cache)

                # Test no face detected
                mock_locations.return_value = []
                success = self.identifier.add_reference_face("player2", self.face_image)
                self.assertFalse(success)
                self.assertNotIn("player2", self.identifier.face_encodings_cache)

    def test_identify_player(self):
        """Test complete player identification."""
        # Enable face recognition
        self.identifier.use_face_recognition = True

        # Mock OCR and face recognition
        with patch.object(self.identifier, "_recognize_jersey_number") as mock_jersey:
            with patch.object(self.identifier, "_match_face") as mock_face:
                # Set up mock returns
                mock_jersey.return_value = {"number": "23", "confidence": 0.95}
                mock_face.return_value = {"name": "player1", "confidence": 0.85}

                # Test with both methods succeeding
                result = self.identifier.identify_player(
                    self.test_image,
                    np.array([0, 0, 100, 100]),
                    {"player1": np.array([0.1, 0.2, 0.3])},
                )

                self.assertEqual(result["jersey_number"], "23")
                self.assertGreater(result["jersey_confidence"], 0.7)
                self.assertEqual(result["face_match"], "player1")
                self.assertGreater(result["face_confidence"], 0.7)

                # Test with only jersey recognition
                mock_face.return_value = None
                result = self.identifier.identify_player(
                    self.test_image,
                    np.array([0, 0, 100, 100]),
                    {"player1": np.array([0.1, 0.2, 0.3])},
                )

                self.assertEqual(result["jersey_number"], "23")
                self.assertGreater(result["jersey_confidence"], 0.7)
                self.assertIsNone(result["face_match"])
                self.assertEqual(result["face_confidence"], 0.0)

                # Test with only face recognition
                mock_jersey.return_value = None
                mock_face.return_value = {"name": "player1", "confidence": 0.85}
                result = self.identifier.identify_player(
                    self.test_image,
                    np.array([0, 0, 100, 100]),
                    {"player1": np.array([0.1, 0.2, 0.3])},
                )

                self.assertIsNone(result["jersey_number"])
                self.assertEqual(result["jersey_confidence"], 0.0)
                self.assertEqual(result["face_match"], "player1")
                self.assertGreater(result["face_confidence"], 0.7)

    def test_get_identification_info(self):
        """Test getting identification information."""
        info = self.identifier.get_identification_info()

        self.assertIn("tracking_mode", info)
        self.assertIn("ocr_available", info)
        self.assertIn("face_recognition_enabled", info)
        self.assertIn("gpu_available", info)
        self.assertIn("confidence_threshold", info)
        self.assertIn("cached_faces", info)
        self.assertEqual(info["confidence_threshold"], 0.7)
        self.assertEqual(info["cached_faces"], [])
