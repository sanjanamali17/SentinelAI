import requests
import json
from PIL import Image
import numpy as np
import io

def create_test_image():
    """Create a simple test image"""
    # Create a simple 100x100 test image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

def test_upload():
    try:
        # Test upload with a simple image
        test_img = create_test_image()
        
        files = {'file': ('test.png', test_img, 'image/png')}
        response = requests.post("http://localhost:8000/upload-official", files=files)
        
        print(f"Upload Status: {response.status_code}")
        print(f"Upload Response: {response.json()}")
        
        # Test status after upload
        status_response = requests.get("http://localhost:8000/status")
        print(f"Status after upload: {status_response.json()}")
        
        # Test analysis with the same image
        test_img2 = create_test_image()
        files2 = {'file': ('test2.png', test_img2, 'image/png')}
        analysis_response = requests.post("http://localhost:8000/check-image", files=files2)
        
        print(f"Analysis Status: {analysis_response.status_code}")
        print(f"Analysis Response: {analysis_response.json()}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_upload()
