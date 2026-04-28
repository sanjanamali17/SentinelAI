from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io
import numpy as np
import base64
from typing import Optional
import json
import os

app = FastAPI(title="SentinelAI API", description="Sports Media Integrity & Piracy Detection")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, update to specific frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and official embedding
model = None
processor = None
official_embedding = None

def load_model():
    """Load CLIP model and processor"""
    global model, processor
    if model is None:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()

def get_image_embedding(image_data: bytes) -> np.ndarray:
    """Generate CLIP embedding for an image"""
    load_model()
    
    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_data))
    
    # Process image and get embedding
    inputs = processor(images=image, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        # Get image features - this returns BaseModelOutputWithPooling
        output = model.get_image_features(**inputs)
        # Extract the pooler_output which is the CLIP embedding (shape: [1, 512])
        image_features = output.pooler_output
        
        # Convert to numpy and flatten
        embedding = image_features.cpu().numpy().flatten()
        # Normalize using numpy
        embedding = embedding / np.linalg.norm(embedding)
    
    return embedding

def calculate_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Calculate cosine similarity between two embeddings"""
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return float(similarity * 100)  # Convert to percentage

def classify_risk(similarity_score: float) -> str:
    """Classify risk level based on similarity score"""
    if similarity_score >= 80:
        return "HIGH"
    elif similarity_score >= 50:
        return "SUSPICIOUS"
    else:
        return "SAFE"

@app.get("/")
async def root():
    return {"message": "SentinelAI API is running", "status": "active"}

@app.post("/upload-official")
async def upload_official(file: UploadFile = File(...)):
    """Upload and store official sports image embedding"""
    global official_embedding
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        # Generate embedding
        official_embedding = get_image_embedding(image_data)
        
        return {
            "status": "success",
            "message": "Official image uploaded and fingerprinted successfully",
            "embedding_shape": official_embedding.shape,
            "fingerprint_id": f"FP_{hash(base64.b64encode(image_data)) % 1000000:06d}"
        }
        
    except Exception as e:
        import traceback
        error_detail = f"Error processing official image: {str(e)}\nTraceback: {traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_detail)

@app.post("/check-image")
async def check_image(file: UploadFile = File(...)):
    """Check suspected image against official embedding"""
    global official_embedding
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Check if official image exists
        if official_embedding is None:
            raise HTTPException(status_code=400, detail="No official image uploaded yet")
        
        # Read image data
        image_data = await file.read()
        
        # Generate embedding
        suspected_embedding = get_image_embedding(image_data)
        
        # Calculate similarity
        similarity_score = calculate_cosine_similarity(suspected_embedding, official_embedding)
        
        # Classify risk
        risk_level = classify_risk(similarity_score)
        
        # Determine if it's a match
        is_match = similarity_score >= 50  # Threshold for match
        
        return {
            "similarity": round(similarity_score, 1),
            "risk": risk_level,
            "match": is_match,
            "analysis": {
                "embedding_compared": True,
                "risk_classification": risk_level,
                "similarity_threshold": 50.0,
                "propagation_depth_score": min(95, similarity_score + 10)  # Simulated propagation score
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing image: {str(e)}")

@app.get("/status")
async def get_status():
    """Get system status"""
    global official_embedding
    
    return {
        "system_status": "active",
        "model_loaded": model is not None,
        "official_image_uploaded": official_embedding is not None,
        "supported_formats": ["JPEG", "PNG", "WebP", "BMP"],
        "api_version": "1.0.0"
    }

@app.post("/reset")
async def reset_system():
    """Reset the system (clear official embedding)"""
    global official_embedding
    official_embedding = None
    return {"status": "success", "message": "System reset successfully"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
