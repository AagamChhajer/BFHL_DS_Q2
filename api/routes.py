from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import io

class ImageRequest(BaseModel):
    image_base64: str

@app.post("/api/get-lab-tests")
async def get_lab_tests(request: ImageRequest):
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        # Process image
        results = processor.process_report(image)
        
        return {
            "is_success": True,
            "data": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))