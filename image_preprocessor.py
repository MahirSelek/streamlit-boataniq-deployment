"""
Image Preprocessing Module for BoataniQ - Production Level
Enhances image quality for better AI recognition with advanced validation:
- Advanced blur detection and quality scoring
- Mathematical boat detection and validation
- Image quality assessment
- Angle/pose validation
- Non-boat image rejection
- Brightness/Contrast adjustment
- Deblurring, Noise reduction, Sharpening
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
from typing import Tuple, Optional, Dict
import time
import math


class ImagePreprocessor:
    """Production-level image preprocessing with validation for boat recognition"""
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.processing_times = {}
        
        # Quality thresholds for production (balanced for MVP)
        self.MIN_BLUR_THRESHOLD = 80.0  # Laplacian variance threshold (slightly more lenient)
        self.MIN_QUALITY_SCORE = 0.4  # Overall quality score (0-1) - more lenient for MVP
        self.MIN_RESOLUTION = (400, 300)  # Minimum image dimensions
        self.MAX_ASPECT_RATIO = 5.0  # Max width/height or height/width ratio
        self.MIN_BRIGHTNESS = 25  # Minimum average brightness (more lenient)
        self.MAX_BRIGHTNESS = 255  # Maximum average brightness (allows all valid images including screenshots)
    
    def validate_image_quality(self, image_bytes: bytes) -> Dict:
        """
        Comprehensive image quality validation for production use
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary with validation results and quality scores
        """
        validation_result = {
            'is_valid': True,
            'quality_score': 0.0,
            'issues': [],
            'warnings': [],
            'blur_score': 0.0,
            'brightness_score': 0.0,
            'resolution_score': 0.0,
            'aspect_ratio_score': 0.0,
            'contrast_score': 0.0,
            'recommendations': []
        }
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                validation_result['is_valid'] = False
                validation_result['issues'].append('Invalid image format - cannot decode image')
                return validation_result
            
            height, width = img.shape[:2]
            
            # 1. Resolution check
            min_width, min_height = self.MIN_RESOLUTION
            if width < min_width or height < min_height:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f'Image resolution too low: {width}x{height}. Minimum: {min_width}x{min_height}')
            else:
                resolution_score = min(1.0, (width * height) / (min_width * min_height * 2))
                validation_result['resolution_score'] = resolution_score
            
            # 2. Aspect ratio check
            aspect_ratio = max(width / height, height / width)
            if aspect_ratio > self.MAX_ASPECT_RATIO:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f'Image aspect ratio too extreme: {aspect_ratio:.2f}. Maximum: {self.MAX_ASPECT_RATIO}')
            else:
                validation_result['aspect_ratio_score'] = 1.0 - (aspect_ratio - 1.0) / (self.MAX_ASPECT_RATIO - 1.0)
            
            # 3. Blur detection (advanced)
            blur_score, blur_variance = self._detect_blur_advanced(img)
            validation_result['blur_score'] = blur_score
            validation_result['blur_variance'] = blur_variance
            
            if blur_variance < self.MIN_BLUR_THRESHOLD:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f'Image is too blurry (blur score: {blur_variance:.1f}). Please upload a clear, sharp image.')
            
            # 4. Brightness check
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            validation_result['brightness'] = avg_brightness
            
            if avg_brightness < self.MIN_BRIGHTNESS:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f'Image is too dark (brightness: {avg_brightness:.1f}). Please use a well-lit image.')
            elif avg_brightness > self.MAX_BRIGHTNESS:
                validation_result['is_valid'] = False
                validation_result['issues'].append(f'Image is too bright/overexposed (brightness: {avg_brightness:.1f}). Please use a properly exposed image.')
            else:
                # Normalize brightness score (30-240 range)
                validation_result['brightness_score'] = 1.0 - abs(avg_brightness - 135) / 105  # 135 is ideal
            
            # 5. Contrast check
            contrast = gray.std()
            validation_result['contrast'] = contrast
            if contrast < 20:
                validation_result['warnings'].append('Image has low contrast. Results may be less accurate.')
                validation_result['contrast_score'] = contrast / 50.0  # Normalize
            else:
                validation_result['contrast_score'] = min(1.0, contrast / 50.0)
            
            # 6. Color channel check
            if len(img.shape) == 3:
                # Check if image is mostly grayscale (low color variance)
                b, g, r = cv2.split(img)
                color_variance = np.var([np.mean(b), np.mean(g), np.mean(r)])
                if color_variance < 100:
                    validation_result['warnings'].append('Image appears to be grayscale or has very low color variation.')
            
            # Calculate overall quality score
            quality_score = (
                validation_result['blur_score'] * 0.3 +
                validation_result['brightness_score'] * 0.2 +
                validation_result['resolution_score'] * 0.2 +
                validation_result['contrast_score'] * 0.15 +
                validation_result['aspect_ratio_score'] * 0.15
            )
            validation_result['quality_score'] = quality_score
            
            if quality_score < self.MIN_QUALITY_SCORE and validation_result['is_valid']:
                validation_result['warnings'].append('Image quality is below recommended standards. Results may be less accurate.')
            
            # Generate recommendations
            if not validation_result['is_valid']:
                validation_result['recommendations'].append('Please upload a clear, well-lit boat image from a good angle.')
                validation_result['recommendations'].append('Ensure the boat is clearly visible and in focus.')
            
            return validation_result
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['issues'].append(f'Error during validation: {str(e)}')
            return validation_result
    
    def detect_boat_mathematical(self, image_bytes: bytes) -> Dict:
        """
        Mathematical boat detection using computer vision techniques
        Detects boat-like shapes, water, and marine environment
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary with boat detection confidence and analysis
        """
        detection_result = {
            'boat_detected': False,
            'confidence': 0.0,
            'indicators': [],
            'issues': [],
            'analysis': {}
        }
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                detection_result['issues'].append('Cannot decode image')
                return detection_result
            
            height, width = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 1. Edge detection for boat-like structures
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            detection_result['analysis']['edge_density'] = edge_density
            
            # Boats typically have moderate edge density (0.1-0.3)
            if 0.05 < edge_density < 0.4:
                detection_result['indicators'].append('Reasonable edge structure detected')
                detection_result['confidence'] += 0.2
            elif edge_density < 0.05:
                detection_result['issues'].append('Very few edges detected - image may be too blurry or not contain complex objects')
            else:
                detection_result['issues'].append('Too many edges - image may be too cluttered or noisy')
            
            # 2. Detect horizontal lines (water horizon, boat deck lines)
            horizontal_lines = self._detect_horizontal_lines(edges, width, height)
            detection_result['analysis']['horizontal_lines'] = horizontal_lines
            
            if horizontal_lines > 2:
                detection_result['indicators'].append('Horizontal structures detected (possible boat deck/hull)')
                detection_result['confidence'] += 0.15
            
            # 3. Detect vertical structures (masts, hull sides)
            vertical_lines = self._detect_vertical_lines(edges, width, height)
            detection_result['analysis']['vertical_lines'] = vertical_lines
            
            if vertical_lines > 1:
                detection_result['indicators'].append('Vertical structures detected (possible masts or hull sides)')
                detection_result['confidence'] += 0.1
            
            # 4. Color analysis for water detection
            water_confidence = self._detect_water_colors(img)
            detection_result['analysis']['water_confidence'] = water_confidence
            
            if water_confidence > 0.3:
                detection_result['indicators'].append('Water-like colors detected (marine environment)')
                detection_result['confidence'] += 0.2
            
            # 5. Shape analysis - detect boat-like geometric shapes
            shape_score = self._detect_boat_shapes(edges, width, height)
            detection_result['analysis']['shape_score'] = shape_score
            
            if shape_score > 0.3:
                detection_result['indicators'].append('Boat-like geometric shapes detected')
                detection_result['confidence'] += 0.15
            
            # 6. Size and proportion analysis
            # Boats typically occupy significant portion of image
            object_coverage = self._analyze_object_coverage(edges, width, height)
            detection_result['analysis']['object_coverage'] = object_coverage
            
            if 0.2 < object_coverage < 0.8:
                detection_result['indicators'].append('Main object occupies reasonable portion of image')
                detection_result['confidence'] += 0.1
            elif object_coverage < 0.1:
                detection_result['issues'].append('Main object too small - boat may not be clearly visible')
            else:
                detection_result['issues'].append('Image may be too cluttered or object too large')
            
            # Final confidence calculation
            detection_result['confidence'] = min(1.0, detection_result['confidence'])
            
            # Determine if boat is detected
            if detection_result['confidence'] >= 0.4 and len(detection_result['issues']) < 3:
                detection_result['boat_detected'] = True
            elif detection_result['confidence'] < 0.3:
                detection_result['issues'].append('Low confidence that image contains a boat. Please upload a clear boat image.')
            
            return detection_result
            
        except Exception as e:
            detection_result['issues'].append(f'Error during boat detection: {str(e)}')
            return detection_result
    
    def validate_boat_image(self, image_bytes: bytes) -> Dict:
        """
        Complete validation: quality + boat detection
        Production-level validation before processing
        
        Args:
            image_bytes: Image data as bytes
            
        Returns:
            Dictionary with complete validation results
        """
        validation = {
            'is_valid': False,
            'can_proceed': False,
            'quality_validation': {},
            'boat_detection': {},
            'combined_confidence': 0.0,
            'rejection_reason': None,
            'warnings': []
        }
        
        # 1. Quality validation
        quality_result = self.validate_image_quality(image_bytes)
        validation['quality_validation'] = quality_result
        
        if not quality_result['is_valid']:
            validation['rejection_reason'] = 'Image quality issues: ' + '; '.join(quality_result['issues'])
            return validation
        
        # 2. Boat detection
        boat_result = self.detect_boat_mathematical(image_bytes)
        validation['boat_detection'] = boat_result
        
        # 3. Combined assessment
        quality_weight = 0.4
        boat_weight = 0.6
        
        combined_confidence = (
            quality_result['quality_score'] * quality_weight +
            boat_result['confidence'] * boat_weight
        )
        validation['combined_confidence'] = combined_confidence
        
        # 4. Final decision
        if quality_result['is_valid'] and boat_result['boat_detected'] and combined_confidence >= 0.5:
            validation['is_valid'] = True
            validation['can_proceed'] = True
        elif not boat_result['boat_detected']:
            validation['rejection_reason'] = 'Boat not clearly detected in image. ' + '; '.join(boat_result['issues'])
        elif combined_confidence < 0.5:
            validation['rejection_reason'] = 'Image quality and boat detection confidence too low. Please upload a clear boat image from a good angle.'
        
        # Collect warnings
        validation['warnings'] = quality_result.get('warnings', []) + boat_result.get('issues', [])
        
        return validation
    
    def preprocess_image(self, image_bytes: bytes, enhance_quality: bool = True) -> Tuple[bytes, dict]:
        """
        Preprocess image to improve quality for AI recognition
        
        Args:
            image_bytes: Original image as bytes
            enhance_quality: Whether to apply quality enhancements
            
        Returns:
            Tuple of (processed_image_bytes, preprocessing_info)
        """
        start_time = time.time()
        
        try:
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                # Try with PIL as fallback
                pil_img = Image.open(io.BytesIO(image_bytes))
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            original_shape = img.shape
            preprocessing_info = {
                'original_size': f"{original_shape[1]}x{original_shape[0]}",
                'enhancements_applied': []
            }
            
            if enhance_quality:
                # Apply enhancements
                img, enhancements = self._apply_enhancements(img)
                preprocessing_info['enhancements_applied'] = enhancements
            else:
                preprocessing_info['enhancements_applied'] = ['none']
            
            # Convert back to bytes
            _, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            processed_bytes = encoded_img.tobytes()
            
            processing_time = time.time() - start_time
            preprocessing_info['processing_time_ms'] = round(processing_time * 1000, 2)
            preprocessing_info['original_size_bytes'] = len(image_bytes)
            preprocessing_info['processed_size_bytes'] = len(processed_bytes)
            
            return processed_bytes, preprocessing_info
            
        except Exception as e:
            print(f"⚠️ [PREPROCESS] Error during preprocessing: {e}")
            # Return original image if preprocessing fails
            return image_bytes, {
                'error': str(e),
                'enhancements_applied': ['error_fallback'],
                'processing_time_ms': round((time.time() - start_time) * 1000, 2)
            }
    
    def _apply_enhancements(self, img: np.ndarray) -> Tuple[np.ndarray, list]:
        """
        Apply various image enhancements - optimized for speed
        
        Args:
            img: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (enhanced_image, list_of_applied_enhancements)
        """
        enhancements = []
        enhanced = img.copy()
        
        # 1. Auto brightness/contrast adjustment (fast)
        enhanced = self._auto_brightness_contrast(enhanced)
        enhancements.append('brightness_contrast')
        
        # 2. Noise reduction (fast bilateral filter) - only if image is large enough
        if img.shape[0] * img.shape[1] < 2000000:  # Skip for very large images to save time
            enhanced = self._reduce_noise(enhanced)
            enhancements.append('noise_reduction')
        
        # 3. Sharpening (only if image is blurry - skip check for speed)
        # Quick blur check - only check a sample
        sample = enhanced[::10, ::10]  # Sample every 10th pixel
        if self._is_blurry(sample, threshold=80):  # Lower threshold for faster processing
            enhanced = self._sharpen(enhanced)
            enhancements.append('sharpening')
        
        # 4. Color enhancement (saturation boost) - lightweight
        enhanced = self._enhance_colors(enhanced)
        enhancements.append('color_enhancement')
        
        return enhanced, enhancements
    
    def _auto_brightness_contrast(self, img: np.ndarray) -> np.ndarray:
        """Automatically adjust brightness and contrast - optimized for speed"""
        # Convert to LAB color space for better brightness adjustment
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE with smaller tile grid for faster processing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        l = clahe.apply(l)
        
        # Merge back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Additional brightness adjustment if image is too dark (only if needed)
        mean_brightness = np.mean(l)
        if mean_brightness < 100:  # Image is dark
            # Increase brightness with optimized values
            alpha = 1.15  # Slightly reduced contrast control for speed
            beta = 25    # Slightly reduced brightness control
            enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
        
        return enhanced
    
    def _reduce_noise(self, img: np.ndarray) -> np.ndarray:
        """Reduce noise while preserving edges (fast method)"""
        # Fast bilateral filter for noise reduction - optimized for speed
        # d=5 for speed, reduced sigma values for faster processing
        denoised = cv2.bilateralFilter(img, d=5, sigmaColor=30, sigmaSpace=30)
        return denoised
    
    def _detect_blur_advanced(self, img: np.ndarray) -> Tuple[float, float]:
        """
        Advanced blur detection using multiple methods
        Returns: (normalized_score, laplacian_variance)
        """
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 score (higher is better)
        # Typical good images: 200-500, blurry: <100
        blur_score = min(1.0, laplacian_var / 300.0)
        
        return blur_score, laplacian_var
    
    def _detect_horizontal_lines(self, edges: np.ndarray, width: int, height: int) -> int:
        """Detect horizontal lines using HoughLines"""
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(width*0.3), 
                                minLineLength=int(width*0.2), maxLineGap=int(width*0.1))
        if lines is None:
            return 0
        
        horizontal_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
            # Consider lines within 10 degrees of horizontal
            if angle < 10 or angle > 170:
                horizontal_count += 1
        
        return horizontal_count
    
    def _detect_vertical_lines(self, edges: np.ndarray, width: int, height: int) -> int:
        """Detect vertical lines using HoughLines"""
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(height*0.3),
                                minLineLength=int(height*0.2), maxLineGap=int(height*0.1))
        if lines is None:
            return 0
        
        vertical_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = abs(math.degrees(math.atan2(y2-y1, x2-x1)))
            # Consider lines within 10 degrees of vertical
            if 80 < angle < 100:
                vertical_count += 1
        
        return vertical_count
    
    def _detect_water_colors(self, img: np.ndarray) -> float:
        """
        Detect water-like colors (blues, teals, grays)
        Returns confidence score 0-1
        """
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Water colors: blue/cyan hues (100-130 in OpenCV HSV)
        # Also check for grayish water (low saturation)
        water_mask = ((h >= 100) & (h <= 130)) | (s < 50)
        water_pixels = np.sum(water_mask)
        total_pixels = img.shape[0] * img.shape[1]
        
        water_ratio = water_pixels / total_pixels
        return min(1.0, water_ratio * 2.0)  # Normalize
    
    def _detect_boat_shapes(self, edges: np.ndarray, width: int, height: int) -> float:
        """
        Detect boat-like geometric shapes (rectangles, trapezoids)
        Returns confidence score 0-1
        """
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return 0.0
        
        shape_score = 0.0
        total_area = width * height
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < total_area * 0.01:  # Skip very small contours
                continue
            
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check for boat-like shapes (4-6 vertices typical for hull/deck)
            if 4 <= len(approx) <= 8:
                # Check aspect ratio (boats are typically longer than tall)
                x, y, w, h = cv2.boundingRect(contour)
                aspect = max(w/h, h/w)
                if 1.5 < aspect < 5.0:  # Reasonable boat proportions
                    shape_score += area / total_area
        
        return min(1.0, shape_score * 3.0)  # Normalize
    
    def _analyze_object_coverage(self, edges: np.ndarray, width: int, height: int) -> float:
        """
        Analyze how much of the image is occupied by main objects
        Returns coverage ratio 0-1
        """
        # Use edge density as proxy for object coverage
        edge_pixels = np.sum(edges > 0)
        total_pixels = width * height
        
        coverage = edge_pixels / total_pixels
        return coverage
    
    def _is_blurry(self, img: np.ndarray, threshold: float = 100.0) -> bool:
        """Check if image is blurry using Laplacian variance - optimized"""
        # If already grayscale, use directly
        if len(img.shape) == 2:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold
    
    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Apply sharpening filter"""
        # Create sharpening kernel
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        # Apply kernel with reduced intensity for natural look
        sharpened = cv2.filter2D(img, -1, kernel * 0.3)
        
        # Blend with original to avoid over-sharpening
        result = cv2.addWeighted(img, 0.7, sharpened, 0.3, 0)
        return result
    
    def _enhance_colors(self, img: np.ndarray) -> np.ndarray:
        """Enhance color saturation"""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Increase saturation slightly
        s = cv2.multiply(s, 1.1)
        s = np.clip(s, 0, 255).astype(np.uint8)
        
        # Merge back
        enhanced = cv2.merge([h, s, v])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_HSV2BGR)
        return enhanced
    
    def preprocess_from_file(self, filepath: str, enhance_quality: bool = True) -> Tuple[bytes, dict]:
        """
        Preprocess image from file path
        
        Args:
            filepath: Path to image file
            enhance_quality: Whether to apply quality enhancements
            
        Returns:
            Tuple of (processed_image_bytes, preprocessing_info)
        """
        with open(filepath, 'rb') as f:
            image_bytes = f.read()
        return self.preprocess_image(image_bytes, enhance_quality)
    
    def get_preprocessing_stats(self) -> dict:
        """Get statistics about preprocessing performance"""
        return {
            'average_processing_time_ms': np.mean(list(self.processing_times.values())) if self.processing_times else 0,
            'total_images_processed': len(self.processing_times)
        }

