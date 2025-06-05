import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

class VideoProcessor:
    def __init__(self, fps_extraction: int = 2):
        """
        Extract keyframes from videos for fashion item detection
        
        Args:
            fps_extraction: Frames per second to extract (default 2 as per hackathon specs)
        """
        self.fps_extraction = fps_extraction
    
    def extract_keyframes(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Extract keyframes from video at specified FPS
        Returns frame data with timestamp and frame number as required by hackathon
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 30  # Default fallback
            
        frame_interval = max(1, int(fps / self.fps_extraction))
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB for consistency with CLIP
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append({
                    'frame': rgb_frame,
                    'timestamp': frame_count / fps,
                    'frame_number': len(frames)
                })
            frame_count += 1
        
        cap.release()
        return frames
