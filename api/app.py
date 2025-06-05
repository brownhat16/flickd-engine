from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import sys
import redis
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main_engine import FlickdEngine

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Flickd Smart Tagging & Vibe Classification Engine",
    description="AI-powered fashion item detection and vibe classification",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Redis
try:
    redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
    redis_client.ping()
    logger.info("Redis connected successfully")
except Exception as e:
    logger.warning(f"Redis connection failed: {e}")
    redis_client = None

# Initialize the engine
try:
    engine = FlickdEngine(
        catalog_path="data",
        vibes_json_path="data/vibes.json"
    )
    logger.info("✅ Flickd Engine initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing engine: {e}")
    engine = None

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Flickd Smart Tagging & Vibe Classification Engine",
        "version": "1.0.0",
        "status": "active" if engine else "error - engine not initialized",
        "endpoints": {
            "process_video": "/process_video",
            "health": "/health",
            "vibes": "/vibes",
            "docs": "/docs"
        }
    }

@app.post("/process_video")
async def process_video(
    video: UploadFile = File(...),
    caption: str = Form(None)
):
    """Process video and return tagging results in Flickd hackathon format"""
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not available")
    
    # Validate file type
    if not video.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    # Create temp directory if it doesn't exist
    temp_dir = Path("/app/temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=temp_dir) as tmp_file:
        try:
            content = await video.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
            
            logger.info(f"Processing video: {video.filename}")
            
            # Process video using Flickd engine
            result = engine.process_video(tmp_path, caption)
            
            logger.info(f"Processing completed for {video.filename}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video {video.filename}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    health_status = {
        "status": "healthy" if engine else "unhealthy",
        "engine_initialized": engine is not None,
        "redis_available": redis_client is not None,
        "components": {
            "video_processor": True,
            "fashion_detector": True,
            "product_matcher": True,
            "vibe_classifier": True
        }
    }
    
    # Check Redis
    if redis_client:
        try:
            redis_client.ping()
            health_status["components"]["redis"] = True
        except:
            health_status["components"]["redis"] = False
            health_status["status"] = "degraded"
    
    return health_status

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
