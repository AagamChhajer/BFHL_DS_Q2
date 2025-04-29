import cv2
from app import LabReportProcessor

def test_single_image(image_path):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # Initialize processor
    processor = LabReportProcessor()
    
    # Process image
    try:
        results = processor.process_report(image)
        
        # Print results
        print("\nExtracted Lab Tests:")
        print("-" * 50)
        for test in results:
            print(f"Test Name: {test['test_name']}")
            print(f"Value: {test['test_value']}")
            print(f"Reference Range: {test['bio_reference_range']}")
            print(f"Out of Range: {test['lab_test_out_of_range']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    # Replace with your image path
    image_path = r"C:\Users\HP\Downloads\bajaj_ds\bajaj_q2\lbmaske\MUM-0425-PA-0004300_REPORTS_27-04-2025_0134-06_PM@G.pdf_page_13.png"

    test_single_image(image_path)