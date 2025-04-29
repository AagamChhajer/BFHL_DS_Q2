import re
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional


class LabTableDetector:
    """
    Advanced detection of tables in lab reports using computer vision techniques
    """
    
    def __init__(self):
        self.min_line_length = 100
        self.max_line_gap = 10
    
    def detect_lines(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect horizontal and vertical lines in the image
        
        Args:
            image: Grayscale image
            
        Returns:
            Tuple of horizontal and vertical lines
        """
        # Create binary image
        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
        
        return horizontal_lines, vertical_lines
    
    def find_table_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Find regions in the image that contain tables
        
        Args:
            image: Grayscale image
            
        Returns:
            List of bounding boxes for table regions
        """
        # Detect lines
        horizontal_lines, vertical_lines = self.detect_lines(image)
        
        # Combine lines
        combined = cv2.add(horizontal_lines, vertical_lines)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours to find tables
        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter small regions
            if w > 100 and h > 100:
                table_regions.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h
                })
        
        return table_regions


class LabTestNormalizer:
    """
    Normalizes and enhances extracted lab test data
    """
    
    def __init__(self):
        # Common test name variations and their standardized names
        self.test_name_map = {
            "hgb": "Hemoglobin",
            "hb": "Hemoglobin",
            "hemoglobin": "Hemoglobin",
            "wbc": "White Blood Cell Count",
            "white blood cell": "White Blood Cell Count",
            "leukocytes": "White Blood Cell Count",
            "rbc": "Red Blood Cell Count",
            "red blood cell": "Red Blood Cell Count",
            "erythrocytes": "Red Blood Cell Count",
            "plt": "Platelets",
            "platelets": "Platelets",
            "thrombocytes": "Platelets",
            "glu": "Glucose",
            "fbs": "Fasting Blood Sugar",
            "glucose": "Glucose",
            "chol": "Cholesterol",
            "total cholesterol": "Cholesterol",
            "hdl": "HDL Cholesterol",
            "ldl": "LDL Cholesterol",
            "trig": "Triglycerides",
            "triglycerides": "Triglycerides",
            "na": "Sodium",
            "sodium": "Sodium",
            "k": "Potassium",
            "potassium": "Potassium",
            "cl": "Chloride",
            "chloride": "Chloride",
            "ca": "Calcium",
            "calcium": "Calcium",
            "mg": "Magnesium",
            "magnesium": "Magnesium",
            "creat": "Creatinine",
            "creatinine": "Creatinine",
            "bun": "Blood Urea Nitrogen",
            "urea": "Blood Urea Nitrogen",
            "alt": "Alanine Transaminase",
            "sgpt": "Alanine Transaminase",
            "ast": "Aspartate Transaminase",
            "sgot": "Aspartate Transaminase",
            "alp": "Alkaline Phosphatase",
            "tbil": "Total Bilirubin",
            "total bilirubin": "Total Bilirubin",
            "alb": "Albumin",
            "albumin": "Albumin",
            "tsh": "Thyroid Stimulating Hormone",
            "t3": "Triiodothyronine",
            "t4": "Thyroxine",
            "hba1c": "Hemoglobin A1c",
            "a1c": "Hemoglobin A1c",
            "vit d": "Vitamin D",
            "25-oh vit d": "Vitamin D",
            "vit b12": "Vitamin B12",
            "cobalamin": "Vitamin B12",
            "fe": "Iron",
            "iron": "Iron",
            "ferritin": "Ferritin"
        }
        
        # Common units for different tests
        self.test_units = {
            "Hemoglobin": "g/dL",
            "White Blood Cell Count": "10^3/µL",
            "Red Blood Cell Count": "10^6/µL",
            "Platelets": "10^3/µL",
            "Glucose": "mg/dL",
            "Cholesterol": "mg/dL",
            "HDL Cholesterol": "mg/dL",
            "LDL Cholesterol": "mg/dL",
            "Triglycerides": "mg/dL",
            "Sodium": "mEq/L",
            "Potassium": "mEq/L",
            "Chloride": "mEq/L",
            "Calcium": "mg/dL",
            "Magnesium": "mg/dL",
            "Creatinine": "mg/dL",
            "Blood Urea Nitrogen": "mg/dL",
            "Alanine Transaminase": "U/L",
            "Aspartate Transaminase": "U/L",
            "Alkaline Phosphatase": "U/L",
            "Total Bilirubin": "mg/dL",
            "Albumin": "g/dL",
            "Thyroid Stimulating Hormone": "µIU/mL",
            "Triiodothyronine": "ng/dL",
            "Thyroxine": "µg/dL",
            "Hemoglobin A1c": "%",
            "Vitamin D": "ng/mL",
            "Vitamin B12": "pg/mL",
            "Iron": "µg/dL",
            "Ferritin": "ng/mL"
        }
        
        # Common reference ranges for tests
        self.common_reference_ranges = {
            "Hemoglobin": {"male": "13.5-17.5", "female": "12.0-15.5"},
            "White Blood Cell Count": "4.5-11.0",
            "Red Blood Cell Count": {"male": "4.5-5.9", "female": "4.0-5.2"},
            "Platelets": "150-450",
            "Glucose": "70-99",
            "Cholesterol": "<200",
            "HDL Cholesterol": ">40",
            "LDL Cholesterol": "<100",
            "Triglycerides": "<150",
            "Sodium": "135-145",
            "Potassium": "3.5-5.0",
            "Chloride": "98-107",
            "Calcium": "8.5-10.5",
            "Magnesium": "1.7-2.2",
            "Creatinine": {"male": "0.7-1.3", "female": "0.6-1.1"},
            "Blood Urea Nitrogen": "7-20",
            "Alanine Transaminase": {"male": "7-55", "female": "7-45"},
            "Aspartate Transaminase": {"male": "8-48", "female": "8-43"},
            "Alkaline Phosphatase": "44-147",
            "Total Bilirubin": "0.1-1.2",
            "Albumin": "3.5-5.0",
            "Thyroid Stimulating Hormone": "0.4-4.0",
            "Triiodothyronine": "80-200",
            "Thyroxine": "5.0-12.0",
            "Hemoglobin A1c": "4.0-5.6",
            "Vitamin D": "30-100",
            "Vitamin B12": "200-900",
            "Iron": {"male": "65-175", "female": "50-170"},
            "Ferritin": {"male": "20-250", "female": "10-120"}
        }
    
    def normalize_test_name(self, test_name: str) -> str:
        """
        Normalize test name to standard form
        
        Args:
            test_name: Raw test name
            
        Returns:
            Standardized test name
        """
        # Convert to lowercase for matching
        test_lower = test_name.lower()
        
        # Check for direct match
        for key, value in self.test_name_map.items():
            if key == test_lower:
                return value
        
        # Check for partial match
        for key, value in self.test_name_map.items():
            if key in test_lower:
                return value
        
        # Return original if no match found
        return test_name
    
    def add_missing_units(self, test_name: str, test_value: str) -> Tuple[str, str]:
        """
        Add units to test value if missing
        
        Args:
            test_name: Standardized test name
            test_value: Test value
            
        Returns:
            Tuple of (value, unit)
        """
        # Check if value already has units
        if any(unit in test_value for unit in ["g/dL", "mg/dL", "U/L", "mEq/L", "%", "ng/mL", "pg/mL", "µg/dL", "10^"]):
            # Split value and unit
            parts = test_value.split()
            if len(parts) > 1:
                return parts[0], parts[1]
            return test_value, ""
        
        # Add standard units if available
        if test_name in self.test_units:
            return test_value, self.test_units[test_name]
        
        return test_value, "N/A"
    
    def suggest_reference_range(self, test_name: str) -> Optional[str]:
        """
        Suggest reference range for test if missing
        
        Args:
            test_name: Standardized test name
            
        Returns:
            Suggested reference range or None
        """
        if test_name in self.common_reference_ranges:
            ref_range = self.common_reference_ranges[test_name]
            
            # Handle gender-specific ranges
            if isinstance(ref_range, dict):
                # Default to the wider range
                values = list(ref_range.values())
                return values[0]
            else:
                return ref_range
                
        return None
    
    def normalize_lab_test(self, lab_test: Dict) -> Dict:
        """
        Normalize lab test data
        
        Args:
            lab_test: Lab test dictionary
            
        Returns:
            Normalized lab test dictionary
        """
        # Normalize test name
        normalized_name = self.normalize_test_name(lab_test["test_name"])
        lab_test["test_name"] = normalized_name
        
        # Add missing units
        if lab_test["test_value"] != "N/A":
            value, unit = self.add_missing_units(normalized_name, lab_test["test_value"])
            lab_test["test_value"] = value
            lab_test["test_unit"] = unit
        else:
            lab_test["test_unit"] = "N/A"
        
        # Suggest reference range if missing or invalid
        if lab_test["bio_reference_range"] == "N/A" or not re.search(r'\d', lab_test["bio_reference_range"]):
            suggested_range = self.suggest_reference_range(normalized_name)
            if suggested_range:
                lab_test["bio_reference_range"] = suggested_range
        
        return lab_test


class ValueExtractor:
    def __init__(self):
        self.value_pattern = re.compile(r'(\d+\.?\d*)\s*([a-zA-Z/%]+)?')
        self.range_pattern = re.compile(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)')
        self.range_pattern_alt = re.compile(r'(\d+\.?\d*)\s*to\s*(\d+\.?\d*)')
        
        # Add specific test mappings for renal function tests
        self.specific_test_mappings = {
            "BLOOD UREA": {
                "name": "Blood Urea",
                "unit": "mg/dl",
                "range": "16-50"
            },
            "BLOOD UREA NITROGEN": {
                "name": "Blood Urea Nitrogen",
                "unit": "mg/dl",
                "range": "5-20"
            },
            "CREATININE": {
                "name": "Creatinine",
                "unit": "mg/dl",
                "range": "0.5-1.5"
            },
            "URIC ACID": {
                "name": "Uric Acid",
                "unit": "mg/dl",
                "range": "2.5-7.2"
            },
            "TOTAL PROTEIN": {
                "name": "Total Protein",
                "unit": "g/dl",
                "range": "6-8"
            },
            "ALBUMIN": {
                "name": "Albumin",
                "unit": "g/dl",
                "range": "3.5-5.0"
            },
            "GLOBULIN": {
                "name": "Globulin",
                "unit": "g/dl",
                "range": "2.5-4.0"
            },
            "A/G RATIO": {
                "name": "A/G Ratio",
                "unit": "ratio",
                "range": "N/A"
            },
            "CALCIUM": {
                "name": "Calcium",
                "unit": "mg/dl",
                "range": "8.4-10.5"
            },
            "INORGANIC PHOSPHORUS": {
                "name": "Inorganic Phosphorus",
                "unit": "mg/dl",
                "range": "2.5-5.0"
            },
            "SERUM SODIUM": {
                "name": "Serum Sodium",
                "unit": "mEq/L",
                "range": "135-150"
            },
            "SERUM POTASSIUM": {
                "name": "Serum Potassium",
                "unit": "mEq/L",
                "range": "3.5-5.5"
            },
            "SERUM CHLORIDE": {
                "name": "Serum Chloride",
                "unit": "mEq/L",
                "range": "96-106"
            }
        }

    def extract_value(self, text: str) -> str:
        """
        Extract numeric value and unit from text
        """
        if not text or text == "N/A":
            return "N/A"
        
        # Try to match the value pattern
        match = self.value_pattern.search(text)
        if match:
            value = match.group(1)
            unit = match.group(2) if match.group(2) else ""
            return f"{value} {unit}".strip()
        return "N/A"

    def extract_reference_range(self, text: str) -> str:
        """
        Extract reference range from text
        """
        if not text or text == "N/A":
            return "N/A"
        
        # Try standard range pattern (e.g., "16-50")
        match = self.range_pattern.search(text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        
        # Try alternative pattern (e.g., "16 to 50")
        match = self.range_pattern_alt.search(text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        
        # Check if test is in specific mappings
        for test_name, mapping in self.specific_test_mappings.items():
            if test_name.lower() in text.lower():
                return mapping["range"]
        
        return "N/A"


class ContextualExtractor:
    """
    Extract lab test data using contextual analysis
    """
    
    def __init__(self):
        self.common_section_headers = [
            "CBC", "Complete Blood Count", "Chemistry Panel", "Lipid Panel",
            "Metabolic Panel", "Basic Metabolic Panel", "Comprehensive Metabolic Panel",
            "Thyroid Panel", "Urinalysis", "Liver Function Tests", "Kidney Function Tests",
            "Hematology", "Chemistry", "Serology", "Immunology"
        ]
        
        self.value_extractor = ValueExtractor()
        self.normalizer = LabTestNormalizer()
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections from report text
        
        Args:
            text: Report text
            
        Returns:
            Dictionary of section name to section text
        """
        sections = {}
        current_section = "General"
        section_text = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line is a section header
            is_header = False
            for header in self.common_section_headers:
                if header.lower() in line.lower() and len(line) < 50:  # Headers are usually short
                    if current_section != "General" or section_text:
                        sections[current_section] = '\n'.join(section_text)
                    current_section = line
                    section_text = []
                    is_header = True
                    break
            
            if not is_header:
                section_text.append(line)
        
        # Add the last section
        if section_text:
            sections[current_section] = '\n'.join(section_text)
        
        return sections
    
    def find_test_groups(self, text: str) -> List[Dict]:
        """
        Find groups of related test information
        
        Args:
            text: Section text
            
        Returns:
            List of test dictionaries
        """
        test_groups = []
        
        # Split by potential test patterns
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
            
            # Check if line contains a test name pattern
            if ':' in line or any(c.isalpha() and c.isupper() for c in line):
                # This could be a test name line
                test_info = {}
                
                # Extract potential test name (part before the colon or first number)
                if ':' in line:
                    test_name = line.split(':', 1)[0].strip()
                    rest = line.split(':', 1)[1].strip()
                else:
                    # Try to separate text from numbers
                    parts = re.split(r'(\d)', line, 1)
                    if len(parts) > 1:
                        test_name = parts[0].strip()
                        rest = parts[1] + ''.join(parts[2:])
                    else:
                        test_name = line
                        rest = ""
                
                # Normalize test name
                test_info["test_name"] = self.normalizer.normalize_test_name(test_name)
                
                # Extract value and reference range from current line
                test_value = self.value_extractor.extract_value(rest)
                ref_range = self.value_extractor.extract_reference_range(rest)
                
                # If value or reference range not found in current line, check next line
                if (not test_value or not ref_range) and i+1 < len(lines):
                    next_line = lines[i+1]
                    if not test_value:
                        test_value = self.value_extractor.extract_value(next_line)
                    if not ref_range:
                        ref_range = self.value_extractor.extract_reference_range(next_line)
                
                # Store extracted information
                test_info["test_value"] = test_value if test_value else "N/A"
                test_info["bio_reference_range"] = ref_range if ref_range else "N/A"
                
                # Add to test groups
                test_groups.append(test_info)
        
        return test_groups
    
    def extract_tests_from_context(self, text: str) -> List[Dict]:
        """
        Extract lab tests using contextual analysis
        
        Args:
            text: Report text
            
        Returns:
            List of normalized lab test dictionaries
        """
        # First try specific renal function test format
        tests = self.extract_renal_function_tests(text)
        
        # If no tests found, fall back to general extraction
        if not tests:
            # Extract sections
            sections = self.extract_sections(text)
            
            # Extract test groups from each section
            all_tests = []
            for section, section_text in sections.items():
                section_tests = self.find_test_groups(section_text)
                all_tests.extend(section_tests)
            
            # Normalize test data
            tests = []
            for test in all_tests:
                if "test_name" in test and test["test_name"]:
                    normalized = self.normalizer.normalize_lab_test(test)
                    tests.append(normalized)
        
        # Calculate out of range values
        calculator = OutOfRangeCalculator()
        for test in tests:
            test["lab_test_out_of_range"] = calculator.is_out_of_range(
                test["test_value"],
                test["bio_reference_range"]
            )
        
        return tests
    
    def extract_renal_function_tests(self, text: str) -> List[Dict]:
        """
        Extract tests specifically from renal function test format
        """
        tests = []
        lines = text.split('\n')
        
        # Find the start of test results
        start_idx = -1
        for i, line in enumerate(lines):
            if "Test Panel: RENAL FUNCTION TEST" in line:
                start_idx = i
                break
        
        if start_idx == -1:
            return []
        
        # Process each line after the header
        for line in lines[start_idx+1:]:
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or line.startswith("Test") or line.startswith("Export"):
                continue
            
            # Try to match the line with our specific pattern
            parts = line.split()
            if len(parts) >= 4:
                test_name = " ".join(parts[:-3])
                value = parts[-3]
                range_val = parts[-2]
                unit = parts[-1]
                
                if test_name in self.value_extractor.specific_test_mappings:
                    mapping = self.value_extractor.specific_test_mappings[test_name]
                    test = {
                        "test_name": mapping["name"],
                        "test_value": value,
                        "bio_reference_range": mapping["range"],
                        "test_unit": mapping["unit"],
                        "lab_test_out_of_range": False  # Will be calculated later
                    }
                    tests.append(test)
        
        return tests


class OutOfRangeCalculator:
    """
    Advanced calculation of whether values are out of range
    """
    
    def __init__(self):
        self.comparison_patterns = [
            # Extract numeric value with optional < or > symbol
            re.compile(r'(?:<|>)?\s*(\d+\.?\d*)'),
            # Extract numeric value with optional unit
            re.compile(r'(?:<|>)?\s*(\d+\.?\d*)\s*(?:mg/dL|g/dL|U/L|µg/dL|ng/mL|%|mmol/L|µIU/mL|pg/mL|mcg/L|mEq/L)?')
        ]
        
        self.range_extraction_pattern = re.compile(r'(\d+\.?\d*)\s*(?:-|to|–)\s*(\d+\.?\d*)')
    
    def extract_numeric_value(self, value_str: str) -> Optional[float]:
        """
        Extract numeric value from string
        
        Args:
            value_str: Value string
            
        Returns:
            Numeric value or None
        """
        if not value_str or value_str == "N/A":
            return None
            
        for pattern in self.comparison_patterns:
            match = pattern.search(value_str)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None
    
    def extract_range(self, range_str: str) -> Optional[Tuple[float, float]]:
        """
        Extract min and max values from range string
        
        Args:
            range_str: Range string
            
        Returns:
            Tuple of (min, max) or None
        """
        if not range_str or range_str == "N/A":
            return None
            
        match = self.range_extraction_pattern.search(range_str)
        if match:
            try:
                range_min = float(match.group(1))
                range_max = float(match.group(2))
                return (range_min, range_max)
            except ValueError:
                return None
        
        # Handle special case for single threshold values (e.g., "<200")
        if "<" in range_str:
            threshold_match = re.search(r'<\s*(\d+\.?\d*)', range_str)
            if threshold_match:
                try:
                    threshold = float(threshold_match.group(1))
                    return (float('-inf'), threshold)
                except ValueError:
                    return None
        elif ">" in range_str:
            threshold_match = re.search(r'>\s*(\d+\.?\d*)', range_str)
            if threshold_match:
                try:
                    threshold = float(threshold_match.group(1))
                    return (threshold, float('inf'))
                except ValueError:
                    return None
        
        return None
    
    def is_out_of_range(self, value_str: str, range_str: str) -> bool:
        """
        Determine if value is outside reference range
        
        Args:
            value_str: Value string
            range_str: Reference range string
            
        Returns:
            Boolean indicating if value is out of range
        """
        # Extract numeric value
        value = self.extract_numeric_value(value_str)
        if value is None:
            return False
        
        # Extract range
        range_tuple = self.extract_range(range_str)
        if range_tuple is None:
            return False
        
        range_min, range_max = range_tuple
        
        # Special handling for values with < or >
        if "<" in value_str:
            # If value is reported as "less than X", it's only out of range if the lower bound is higher
            actual_value = value
            return actual_value < range_min
        elif ">" in value_str:
            # If value is reported as "greater than X", it's only out of range if the upper bound is lower
            actual_value = value
            return actual_value > range_max
        else:
            # Normal comparison
            return value < range_min or value > range_max