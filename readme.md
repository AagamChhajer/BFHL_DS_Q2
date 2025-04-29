cd \bajaj_q2
uvicorn app:app --reload --port 8000

then Content-Type: application/json

body raw -> {
    "image_path": "C:/Users/HP/Downloads/bajaj_ds/bajaj_q2/your_image.jpg"
}

expected response 
{
  "is_success": true,
  "data": [
    {
      "test_name": "BLOOD UREA",
      "test_value": "24.6",
      "bio_reference_range": "16-50",
      "test_unit": "mg/dl",
      "lab_test_out_of_range": false
    },
    // ... more tests ...
  ]
}

result postman image at \bajaj_q2\output\image.png