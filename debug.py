import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import io

def debug_clip():
    try:
        print("Loading CLIP model...")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        
        print("Creating test image...")
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        print("Processing image...")
        inputs = processor(images=img, return_tensors="pt", padding=True)
        
        print("Getting image features...")
        with torch.no_grad():
            # Try different approaches
            print("Approach 1: get_image_features")
            try:
                output = model.get_image_features(**inputs)
                print(f"Success! Type: {type(output)}")
                print(f"Output attributes: {dir(output)}")
                if hasattr(output, 'last_hidden_state'):
                    print(f"last_hidden_state shape: {output.last_hidden_state.shape}")
                if hasattr(output, 'pooler_output'):
                    print(f"pooler_output shape: {output.pooler_output.shape}")
                # Try to access it as a tensor directly
                try:
                    print(f"Direct tensor shape: {output.shape}")
                except:
                    print("Cannot access shape directly")
            except Exception as e:
                print(f"Error with get_image_features: {e}")
                import traceback
                traceback.print_exc()
            
            print("Approach 2: vision_model")
            try:
                vision_outputs = model.vision_model(**inputs)
                print(f"Vision output type: {type(vision_outputs)}")
                print(f"Vision output attributes: {dir(vision_outputs)}")
                if hasattr(vision_outputs, 'last_hidden_state'):
                    print(f"Vision last_hidden_state shape: {vision_outputs.last_hidden_state.shape}")
                if hasattr(vision_outputs, 'pooler_output'):
                    print(f"Vision pooler_output shape: {vision_outputs.pooler_output.shape}")
            except Exception as e:
                print(f"Error with vision_model: {e}")
                
    except Exception as e:
        print(f"Overall error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_clip()
