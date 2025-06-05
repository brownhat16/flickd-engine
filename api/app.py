from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main_engine import FlickdEngine

# Initialize FastAPI app
app = FastAPI(
    title="Flickd Smart Tagging & Vibe Classification Engine",
    description="AI-powered fashion item detection and vibe classification for Flickd hackathon",
    version="1.0.0"
)

# Initialize the engine with provided datasets
CATALOG_PATH = "data"  # Directory containing images.csv and product_data.csv
VIBES_JSON_PATH = "data/vibes.json"  # Official vibes list

try:
    engine = FlickdEngine(CATALOG_PATH, VIBES_JSON_PATH)
    print("✅ Flickd Engine initialized successfully")
except Exception as e:
    print(f"❌ Error initializing engine: {e}")
    engine = None

@app.post("/process_video")
async def process_video(
    video: UploadFile = File(..., description="Video file to process"),
    caption: str = Form(None, description="Optional caption with hashtags")
):
    """
    Process video and return tagging results in hackathon format
    
    Returns JSON with detected products and classified vibes
    """
    
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    # Validate file type
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        try:
            content = await video.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error saving video: {str(e)}")
    
    try:
        # Process video using Flickd engine
        result = engine.process_video(tmp_path, caption)
        
        # Return results in hackathon format
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if engine else "unhealthy",
        "engine_initialized": engine is not None,
        "hackathon": "Flickd AI Hackathon - Smart Tagging & Vibe Classification"
    }

@app.get("/vibes")
async def get_supported_vibes():
    """Get official Flickd vibes list"""
    return {
        "supported_vibes": [
            "Coquette",
            "Clean Girl", 
            "Cottagecore",
            "Streetcore",
            "Y2K",
            "Boho",
            "Party Glam"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
