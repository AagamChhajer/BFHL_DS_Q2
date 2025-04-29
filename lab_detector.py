import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import pytesseract
import re

class LabReportLayoutAnalyzer:
    """
    Analyze layout of lab reports to identify regions with test information
    """
    
    def __init__(self):
        # Initialize parameters for layout analysis
        self.blur_kernel = (5, 5)
        self.threshold_block_size = 11
        self.canny_low = 50
        self.canny_high = 150
        self.min_line_length = 100
        self.max_line_gap = 10
    
    def preprocess_for_layout(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for layout analysis
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if not already
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)
        
        # Apply adaptive thresholding to get binary image
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, self.threshold_block_size, 2
        )
        
        return binary
    
    def detect_tables(self, binary_image: np.ndarray) -> List[Dict]:
        """
        Detect tables in the binary image
        
        Args:
            binary_image: Binary image
            
        Returns:
            List of table regions as dictionaries
        """
        # Detect lines using Hough transform
        edges = cv2.Canny(binary_image, self.canny_low, self.canny_high)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 100, minLineLength=self.min_line_length, 
            maxLineGap=self.max_line_gap
        )
        
        # If no lines found, return empty list
        if lines is None:
            return []
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle of the line
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            
            # Horizontal lines have angle close to 0 or 180
            if angle < 20 or angle > 160:
                horizontal_lines.append(line[0])
            # Vertical lines have angle close to 90
            elif 70 < angle < 110:
                vertical_lines.append(line[0])
        
        # Find table regions based on intersections of horizontal and vertical lines
        table_regions = []
        
        # If there are enough lines to form tables
        if len(horizontal_lines) > 1 and len(vertical_lines) > 1:
            # Find bounding box of lines
            all_lines = np.vstack((horizontal_lines, vertical_lines))
            x_min = np.min(all_lines[:, [0, 2]])
            y_min = np.min(all_lines[:, [1, 3]])
            x_max = np.max(all_lines[:, [0, 2]])
            y_max = np.max(all_lines[:, [1, 3]])
            
            # Add as a table region
            table_regions.append({
                "x": int(x_min),
                "y": int(y_min),
                "width": int(x_max - x_min),
                "height": int(y_max - y_min)
            })
            
        return table_regions
    
    def detect_text_blocks(self, binary_image: np.ndarray) -> List[Dict]:
        """
        Detect blocks of text in the binary image
        
        Args:
            binary_image: Binary image
            
        Returns:
            List of text block regions as dictionaries
        """
        # Find contours in the binary image
        contours, _ = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter contours to find text blocks
        text_blocks = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter small contours and elongated ones
            if area > 1000 and 0.2 < w/h < 5:
                text_blocks.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                })
        
        return text_blocks
    
    def identify_regions_of_interest(self, image: np.ndarray) -> List[Dict]:
        """
        Identify regions of interest in the lab report image
        
        Args:
            image: Input image
            
        Returns:
            List of regions of interest as dictionaries
        """
        # Preprocess image for layout analysis
        binary = self.preprocess_for_layout(image)
        
        # Detect tables
        tables = self.detect_tables(binary)
        
        # Detect text blocks
        text_blocks = self.detect_text_blocks(binary)
        
        # Combine all regions of interest
        regions = []
        regions.extend(tables)
        regions.extend(text_blocks)
        
        # Merge overlapping regions
        merged_regions = self.merge_overlapping_regions(regions)
        
        return merged_regions
    
    def merge_overlapping_regions(self, regions: List[Dict]) -> List[Dict]:
        """
        Merge overlapping regions
        
        Args:
            regions: List of region dictionaries
            
        Returns:
            List of merged region dictionaries
        """
        if not regions:
            return []
        
        # Sort regions by x coordinate
        regions.sort(key=lambda r: r["x"])
        
        merged = []
        current = regions[0]
        
        for i in range(1, len(regions)):
            next_region = regions[i]
            
            # Check if regions overlap
            if (current["x"] + current["width"] > next_region["x"] and
                current["y"] + current["height"] > next_region["y"] and
                next_region["x"] + next_region["width"] > current["x"] and
                next_region["y"] + next_region["height"] > current["y"]):
                
                # Merge regions
                x1 = min(current["x"], next_region["x"])
                y1 = min(current["y"], next_region["y"])
                x2 = max(current["x"] + current["width"], next_region["x"] + next_region["width"])
                y2 = max(current["y"] + current["height"], next_region["y"] + next_region["height"])
                
                current = {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            else:
                merged.append(current)
                current = next_region
        
        # Add the last region
        merged.append(current)
        
        return merged
    
    def extract_from_regions(self, image: np.ndarray, regions: List[Dict]) -> List[str]:
        """
        Extract text from regions of interest
        
        Args:
            image: Input image
            regions: List of region dictionaries
            
        Returns:
            List of extracted text from each region
        """
        texts = []
        
        for region in regions:
            # Extract region from image
            x = region["x"]
            y = region["y"]
            w = region["width"]
            h = region["height"]
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, image.shape[1] - x)
            h = min(h, image.shape[0] - y)
            
            # Skip invalid regions
            if w <= 0 or h <= 0:
                continue
            
            region_img = image[y:y+h, x:x+w]
            
            # Convert to grayscale if needed
            if len(region_img.shape) == 3:
                region_img = cv2.cvtColor(region_img, cv2.COLOR_BGR2GRAY)
            
            # Apply OCR to extract text
            custom_config = r'--oem 3 --psm 6 -l eng'
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            text = pytesseract.image_to_string(region_img, config=custom_config)
            
            texts.append(text)
        
        return texts
    
    def analyze_layout(self, image: np.ndarray) -> Tuple[List[Dict], List[str]]:
        """
        Analyze layout of lab report and extract text from regions
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (regions, texts)
        """
        # Identify regions of interest
        regions = self.identify_regions_of_interest(image)
        
        # Extract text from regions
        texts = self.extract_from_regions(image, regions)
        
        return regions, texts


class LabReportSegmenter:
    """
    Segment lab report images for improved text extraction
    """
    
    def __init__(self):
        # Initialize parameters for segmentation
        self.min_header_height = 50
        self.header_height_ratio = 0.15  # Top 10% of the image
        self.footer_height_ratio = 0.15  # Bottom 10% of the image
    
    def segment_report(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment lab report into header, body, and footer
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of image segments
        """
        height, width = image.shape[:2]
        
        # Calculate segment boundaries
        header_height = max(self.min_header_height, int(height * self.header_height_ratio))
        footer_start = height - int(height * self.footer_height_ratio)
        
        # Extract segments
        header = image[:header_height, :]
        body = image[header_height:footer_start, :]
        footer = image[footer_start:, :]
        
        return {
            "header": header,
            "body": body,
            "footer": footer,
            "full": image
        }
    
    def extract_patient_info(self, header: np.ndarray) -> Dict[str, str]:
        """
        Extract patient information from header
        
        Args:
            header: Header image
            
        Returns:
            Dictionary of patient information
        """
        # Apply OCR to extract text
        custom_config = r'--oem 3 --psm 6 -l eng'
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        text = pytesseract.image_to_string(header, config=custom_config)
        
        # Parse patient information
        patient_info = {}
        
        # Look for common patient info patterns
        name_match = re.search(r'(?:Patient|Name)[:\s]\s*([A-Za-z\s]+)', text)
        if name_match:
            patient_info["name"] = name_match.group(1).strip()
            
        id_match = re.search(r'(?:ID|MRN)[:\s]\s*([A-Za-z0-9\-]+)', text)
        if id_match:
            patient_info["id"] = id_match.group(1).strip()
            
        dob_match = re.search(r'(?:DOB|Date of Birth)[:\s]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text)
        if dob_match:
            patient_info["dob"] = dob_match.group(1).strip()
            
        date_match = re.search(r'(?:Date|Collection Date)[:\s]\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text)
        if date_match:
            patient_info["date"] = date_match.group(1).strip()

        print("Extracted Patient Info:", patient_info)  # Debugging line
        return patient_info
    
    def process_segments(self, segments: Dict[str, np.ndarray]) -> Dict:
        """
        Process all segments to extract relevant information
        
        Args:
            segments: Dictionary of image segments
            
        Returns:
            Dictionary of extracted information
        """
        # Extract patient info from header
        patient_info = self.extract_patient_info(segments["header"])
        
        # Initialize layout analyzer for body
        layout_analyzer = LabReportLayoutAnalyzer()
        
        # Analyze layout of body
        regions, texts = layout_analyzer.analyze_layout(segments["body"])
        
        # Extract footer information
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        footer_text = pytesseract.image_to_string(segments["footer"])
        
        return {
            "patient_info": patient_info,
            "regions": regions,
            "texts": texts,
            "footer_text": footer_text
        }