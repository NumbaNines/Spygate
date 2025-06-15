# Preprocessing Optimization Test Setup

## Purpose
Systematic testing of preprocessing techniques to find optimal parameters for PaddleOCR on Madden HUD elements.

## Test Samples
- 25 randomly selected screenshots from YOLO training dataset
- Converted to grayscale for consistent testing
- Representative of actual Madden HUD content

## Preprocessing Techniques to Test
1. **Contrast Enhancement** (cv2.convertScaleAbs)
   - Alpha: 0.5 - 3.0 (contrast multiplier)
   - Beta: 0 - 100 (brightness offset)

2. **Gaussian Blur** (cv2.GaussianBlur)
   - Kernel size: 1 - 15 (noise reduction)

3. **Sharpening** (Unsharp masking)
   - Strength: 0.0 - 2.0 (edge enhancement)

4. **Upscaling** (cv2.resize)
   - Scale factor: 1.0 - 5.0 (size increase)
   - Interpolation: CUBIC, LANCZOS4

5. **Gamma Correction** (LUT)
   - Gamma: 0.3 - 3.0 (brightness curve)

6. **Morphological Operations**
   - Opening, Closing, Gradient
   - Kernel sizes: 3x3, 5x5, 7x7

7. **Adaptive Thresholding**
   - ADAPTIVE_THRESH_MEAN_C
   - ADAPTIVE_THRESH_GAUSSIAN_C

## Testing Strategy
1. Test each technique individually on all 25 samples
2. Measure OCR confidence and text detection count
3. Find optimal parameters for each technique
4. Test combinations of best-performing techniques
5. Build final optimized preprocessing pipeline

## Success Metrics
- OCR confidence score (higher is better)
- Number of texts detected (more is better)
- Accuracy of detected text (manual verification)
- Processing speed (faster is better)
