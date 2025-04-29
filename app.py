import os
import cv2
import numpy as np
from typing import List, Dict, Any
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from extraction_enhancer import LabTestNormalizer, ValueExtractor, ContextualExtractor, OutOfRangeCalculator
from lab_detector import LabReportLayoutAnalyzer, LabReportSegmenter
class LabTest(BaseModel):
    test_name: str
    test_value: str
    bio_reference_range: str
    test_unit: str
    lab_test_out_of_range: bool

class LabReportResponse(BaseModel):
    is_success: bool
    data: List[LabTest]
    error_message: Optional[str] = None


app = FastAPI(
    title="Lab Report Test Extractor API",
    description="Extracts lab tests from medical reports without using LLMs",
    version="1.0.0"
)


class LabReportProcessor:
    """
    Main processor for lab report images
    Orchestrates the extraction pipeline
    """
    
    def __init__(self):
        # Initialize components
        self.normalizer = LabTestNormalizer()
        self.value_extractor = ValueExtractor()
        self.contextual_extractor = ContextualExtractor()
        self.out_of_range_calculator = OutOfRangeCalculator()
        self.layout_analyzer = LabReportLayoutAnalyzer()
        self.segmenter = LabReportSegmenter()
        
        # Initialize OCR configuration
        self.ocr_config = r'--oem 3 --psm 6 -l eng'
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image to improve OCR accuracy
        
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
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
        
        return denoised
    
    def extract_tests_from_image(self, image: np.ndarray) -> List[Dict]:
        """
        Extract lab tests from image using multiple approaches
        
        Args:
            image: Input image
            
        Returns:
            List of lab test dictionaries
        """
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Segment report
        segments = self.segmenter.segment_report(preprocessed)
        
        # Process segments
        segment_info = self.segmenter.process_segments(segments)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        # Extract text from full image as fallback
        full_text = pytesseract.image_to_string(preprocessed, config=self.ocr_config)
        print(f"Full Image Text: {full_text}")
        # Combine texts from all segments and regions
        all_texts = segment_info["texts"]
        all_texts.append(full_text)
        combined_text = '\n'.join(all_texts)
        
        # Extract tests using contextual extraction
        tests = self.contextual_extractor.extract_tests_from_context(combined_text)
        print(f"Extracted Tests: {tests}")
        # If no tests found, fallback to simpler extraction
        if not tests:
            tests = self.extract_tests_fallback(combined_text)
        
        # Calculate out of range for each test
        for test in tests:
            test["lab_test_out_of_range"] = self.out_of_range_calculator.is_out_of_range(
                test["test_value"], test["bio_reference_range"]
            )
        
        return tests
    
    def extract_tests_fallback(self, text: str) -> List[Dict]:
        """
        Fallback method for extracting tests from text
        
        Args:
            text: Extracted text
            
        Returns:
            List of lab test dictionaries
        """
        lab_tests = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            
            # Look for patterns like "Test Name: Value (Range)"
            if ':' in line:
                parts = line.split(':', 1)
                test_name = parts[0].strip()
                
                # Skip if test name is too long (likely not a test name)
                if len(test_name) > 50:
                    continue
                
                # Normalize test name
                test_name = self.normalizer.normalize_test_name(test_name)
                
                # Extract value and range
                value_part = parts[1].strip()
                test_value = self.value_extractor.extract_value(value_part)
                ref_range = self.value_extractor.extract_reference_range(value_part)
                
                # Check next line for more info if needed
                if (not test_value or not ref_range) and i+1 < len(lines):
                    next_line = lines[i+1]
                    if not test_value:
                        test_value = self.value_extractor.extract_value(next_line)
                    if not ref_range:
                        ref_range = self.value_extractor.extract_reference_range(next_line)
                
                # Add test if we have at least name and value
                if test_name and test_value:
                    lab_test = {
                        "test_name": test_name,
                        "test_value": test_value if test_value else "N/A",
                        "bio_reference_range": ref_range if ref_range else "N/A"
                    }
                    
                    # Normalize test data
                    lab_test = self.normalizer.normalize_lab_test(lab_test)
                    
                    lab_tests.append(lab_test)
        
        return lab_tests
    
    def process_report(self, image: np.ndarray) -> List[Dict]:
        """
        Process lab report image and extract lab test data
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of lab test dictionaries
        """
        # Extract tests from image
        tests = self.extract_tests_from_image(image)
        
        # Deduplicate and validate tests
        validated_tests = self.validate_and_deduplicate(tests)
        
        return validated_tests
    
    def validate_and_deduplicate(self, tests: List[Dict]) -> List[Dict]:
        """
        Validate and deduplicate extracted tests
        
        Args:
            tests: List of lab test dictionaries
            
        Returns:
            Validated and deduplicated list
        """
        # Remove duplicates based on test name
        seen_tests = {}
        for test in tests:
            test_name = test["test_name"].lower()
            
            # Check if test is already seen
            if test_name in seen_tests:
                # Keep the one with more complete information
                current = seen_tests[test_name]
                
                # Check which test has more complete information
                current_score = (current["test_value"] != "N/A") + (current["bio_reference_range"] != "N/A")
                new_score = (test["test_value"] != "N/A") + (test["bio_reference_range"] != "N/A")
                
                if new_score > current_score:
                    seen_tests[test_name] = test
            else:
                seen_tests[test_name] = test
        
        # Convert back to list
        validated_tests = list(seen_tests.values())
        
        # Ensure all tests have the required fields
        for test in validated_tests:
            if "test_name" not in test or not test["test_name"]:
                test["test_name"] = "Unknown Test"
            if "test_value" not in test or not test["test_value"]:
                test["test_value"] = "N/A"
            if "bio_reference_range" not in test or not test["bio_reference_range"]:
                test["bio_reference_range"] = "N/A"
            if "lab_test_out_of_range" not in test:
                test["lab_test_out_of_range"] = False
        
        return validated_tests


# Initialize processor
processor = LabReportProcessor()


class ImagePathRequest(BaseModel):
    image_path: str

@app.post("/get-lab-tests", response_model=LabReportResponse)
async def extract_lab_tests(request: ImagePathRequest) -> JSONResponse:
    try:
        image_path = request.image_path
        print(f"Processing image from path: {image_path}")
        
        # Validate if file exists
        if not os.path.exists(image_path):
            return JSONResponse(
                status_code=400,
                content={
                    "is_success": False,
                    "data": [],
                    "error_message": f"Image file not found at path: {image_path}"
                }
            )
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return JSONResponse(
                status_code=400,
                content={
                    "is_success": False,
                    "data": [],
                    "error_message": "Could not read image file"
                }
            )
        
        # Process image to extract lab tests
        raw_tests = processor.process_report(image)
        
        # Convert to required format
        formatted_tests = []
        for test in raw_tests:
            # Split value and unit if present
            value = test["test_value"]
            unit = ""
            
            # Extract unit from value if present
            value_parts = value.split()
            if len(value_parts) > 1:
                value = value_parts[0]
                unit = value_parts[1]
            
            formatted_test = {
                "test_name": test["test_name"].upper(),
                "test_value": value,
                "bio_reference_range": test["bio_reference_range"],
                "test_unit": unit or "N/A",
                "lab_test_out_of_range": test["lab_test_out_of_range"]
            }
            formatted_tests.append(formatted_test)
        
        return JSONResponse(
            status_code=200,
            content={
                "is_success": True,
                "data": formatted_tests
            }
        )
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "is_success": False,
                "data": [],
                "error_message": f"Error processing image: {str(e)}"
            }
        )

if __name__ == "__main__":
    # Start FastAPI server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# @app.get("/")
# def read_root():
#     """
#     Root endpoint returning API information
