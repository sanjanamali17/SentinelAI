import requests
import json
from PIL import Image
import numpy as np
import io

def create_test_image():
    """Create a simple test image"""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return img_bytes

def test_upload_detailed():
    try:
        # Test upload with a simple image
        test_img = create_test_image()
        
        files = {'file': ('test.png', test_img, 'image/png')}
        response = requests.post("http://localhost:8000/upload-official", files=files)
        
        print(f"Upload Status: {response.status_code}")
        print(f"Full Response: {response.text}")
        
        if response.status_code != 200:
            print(f"Error details: {response.json()}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_upload_detailed()
