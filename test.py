import requests
import json

def test_backend():
    try:
        # Test basic connection
        response = requests.get("http://localhost:8000/")
        print(f"Root endpoint: {response.status_code} - {response.json()}")
        
        # Test status endpoint
        response = requests.get("http://localhost:8000/status")
        print(f"Status endpoint: {response.status_code} - {response.json()}")
        
        # Test docs endpoint
        response = requests.get("http://localhost:8000/docs")
        print(f"Docs endpoint: {response.status_code}")
        
    except Exception as e:
        print(f"Error testing backend: {e}")

if __name__ == "__main__":
    test_backend()
