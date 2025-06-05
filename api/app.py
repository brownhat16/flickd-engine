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
    title="Flickd Smart Tagging & Vibe Classification API",
    description="AI-powered fashion item detection and vibe classification for short videos",
    version="1.0.0"
)

# Initialize the engine (you'll need to provide actual paths)
CATALOG_PATH = "data/product_catalog.csv"  # Update this path
ASSEMBLYAI_KEY = os.getenv("ASSEMBLYAI_API_KEY")  # Set this environment variable

try:
    engine = FlickdEngine(CATALOG_PATH, ASSEMBLYAI_KEY)
    print("✅ Flickd Engine initialized successfully")
except Exception as e:
    print(f"❌ Error initializing engine: {e}")
    engine = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Flickd Smart Tagging & Vibe Classification API",
        "version": "1.0.0",
        "status": "active" if engine else "error - engine not initialized",
        "endpoints": {
            "process_video": "/process_video",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.post("/process_video")
async def process_video(
    video: UploadFile = File(..., description="Video file to process"),
    caption: str = Form(None, description="Optional caption text")
):
    """
    Process video and return tagging results
    
    Args:
        video: Video file (MP4, AVI, MOV, etc.)
        caption: Optional caption text for vibe classification
        
    Returns:
        JSON with detected products and classified vibes
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
        # Process video
        result = engine.process_video(tmp_path, caption)
        
        # Return results
        return JSONResponse(content=result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.post("/process_batch")
async def process_batch(
    videos: list[UploadFile] = File(..., description="Multiple video files"),
    captions: str = Form(None, description="JSON array of captions (optional)")
):
    """
    Process multiple videos in batch
    
    Args:
        videos: List of video files
        captions: JSON string with array of captions
        
    Returns:
        JSON array with results for each video
    """
    
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    # Parse captions if provided
    caption_list = None
    if captions:
        try:
            import json
            caption_list = json.loads(captions)
        except:
            raise HTTPException(status_code=400, detail="Invalid captions JSON")
    
    temp_files = []
    try:
        # Save all videos temporarily
        for video in videos:
            if not video.content_type.startswith('video/'):
                raise HTTPException(status_code=400, detail=f"File {video.filename} must be a video")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                content = await video.read()
                tmp_file.write(content)
                temp_files.append(tmp_file.name)
        
        # Process all videos
        results = engine.process_batch(temp_files, caption_list)
        
        return JSONResponse(content=results)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing videos: {str(e)}")
    
    finally:
        # Clean up all temporary files
        for tmp_path in temp_files:
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
        "components": {
            "video_processor": True,
            "fashion_detector": True,
            "product_matcher": True,
            "vibe_classifier": True,
            "audio_transcriber": engine.audio_transcriber is not None if engine else False
        }
    }

@app.get("/vibes")
async def get_supported_vibes():
    """Get list of supported fashion vibes"""
    if not engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    return {
        "supported_vibes": list(engine.vibe_classifier.vibe_definitions.keys()),
        "vibe_definitions": engine.vibe_classifier.vibe_definitions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
