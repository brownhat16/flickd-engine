from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from .video_processor import VideoProcessor
from .fashion_detector import FashionDetector
from .product_matcher import ProductMatcher
from .vibe_classifier import VibeClassifier
from .audio_transcriber import AudioTranscriber

class FlickdEngine:
    def __init__(self, catalog_path: str, assemblyai_key: Optional[str] = None):
        """
        Initialize the complete Flickd Smart Tagging & Vibe Classification Engine
        
        Args:
            catalog_path: Path to product catalog CSV
            assemblyai_key: AssemblyAI API key for audio transcription
        """
        # Initialize components
        self.video_processor = VideoProcessor(fps_extraction=2)
        self.fashion_detector = FashionDetector()
        self.product_matcher = ProductMatcher(catalog_path)
        self.vibe_classifier = VibeClassifier()
        
        # Initialize audio transcriber if API key provided
        self.audio_transcriber = None
        if assemblyai_key:
            self.audio_transcriber = AudioTranscriber(assemblyai_key)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_video(self, video_path: str, caption: Optional[str] = None) -> Dict[str, Any]:
        """
        Main processing pipeline for video analysis
        
        Args:
            video_path: Path to video file
            caption: Optional caption text
            
        Returns:
            Dictionary with analysis results
        """
        try:
            video_id = Path(video_path).stem
            self.logger.info(f"Processing video: {video_id}")
            
            # Step 1: Extract frames
            self.logger.info("Extracting keyframes...")
            frames = self.video_processor.extract_keyframes(video_path)
            
            if not frames:
                self.logger.warning("No frames extracted from video")
                return self._empty_result(video_id)
            
            # Step 2: Detect fashion items in frames
            self.logger.info("Detecting fashion items...")
            all_detections = []
            
            for frame_data in frames:
                # Use the enhanced detection method that extracts clothing from persons
                detections = self.fashion_detector.extract_person_clothing(frame_data['frame'])
                
                # Add frame info to detections
                for detection in detections:
                    detection['frame_number'] = frame_data['frame_number']
                    detection['timestamp'] = frame_data['timestamp']
                
                all_detections.extend(detections)
            
            self.logger.info(f"Found {len(all_detections)} fashion item detections")
            
            # Step 3: Match detected items with catalog
            self.logger.info("Matching products...")
            matched_products = []
            
            for detection in all_detections:
                frame_idx = detection['frame_number']
                if frame_idx < len(frames):
                    frame = frames[frame_idx]['frame']
                    
                    match_result = self.product_matcher.match_detected_item(
                        frame, detection['bbox']
                    )
                    
                    product_info = {
                        'type': detection['class'],
                        'confidence': detection['confidence'],
                        'match_type': match_result['match_type'],
                        'matched_product_id': match_result['matched_product_id'],
                        'similarity_score': match_result['similarity_score'],
                        'frame_number': detection['frame_number'],
                        'timestamp': detection['timestamp'],
                        'bbox': detection['bbox']
                    }
                    
                    matched_products.append(product_info)
            
            # Step 4: Transcribe audio if available
            transcript = None
            if self.audio_transcriber:
                self.logger.info("Transcribing audio...")
                transcript = self.audio_transcriber.transcribe_video(video_path)
                if transcript:
                    self.logger.info(f"Transcription: {transcript[:100]}...")
            
            # Step 5: Classify vibes
            self.logger.info("Classifying vibes...")
            text_for_classification = ""
            
            if caption:
                text_for_classification += caption + " "
            if transcript:
                text_for_classification += transcript
            
            vibes = []
            if text_for_classification.strip():
                vibes = self.vibe_classifier.classify_vibes(text_for_classification)
            
            # Step 6: Generate final output
            result = self._generate_output(video_id, vibes, matched_products, transcript, caption)
            
            self.logger.info(f"Processing complete for {video_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
            return self._error_result(Path(video_path).stem, str(e))
    
    def _generate_output(self, video_id: str, vibes: List[str], products: List[Dict], 
                        transcript: Optional[str], caption: Optional[str]) -> Dict[str, Any]:
        """Generate final JSON output according to Flickd specifications"""
        
        # Aggregate and filter products
        aggregated_products = self._aggregate_products(products)
        
        # Build result according to required format
        result = {
            "video_id": video_id,
            "vibes": vibes,
            "products": aggregated_products
        }
        
        # Add optional fields for debugging/analysis
        result["metadata"] = {
            "total_detections": len(products),
            "frames_processed": len(set(p['frame_number'] for p in products)) if products else 0,
            "has_transcript": transcript is not None,
            "has_caption": caption is not None,
            "transcript": transcript,
            "caption": caption
        }
        
        return result
    
    def _aggregate_products(self, products: List[Dict]) -> List[Dict[str, Any]]:
        """
        Aggregate and filter products by type, keeping best matches
        """
        if not products:
            return []
        
        # Group by product type
        type_groups = {}
        for product in products:
            ptype = product['type']
            if ptype not in type_groups:
                type_groups[ptype] = []
            type_groups[ptype].append(product)
        
        # Select best match for each type
        final_products = []
        
        for ptype, group in type_groups.items():
            # Filter out no_match products
            valid_matches = [p for p in group if p['match_type'] != 'no_match']
            
            if not valid_matches:
                continue
            
            # Sort by confidence * similarity score
            best_product = max(valid_matches, 
                             key=lambda x: x['confidence'] * x['similarity_score'])
            
            # Only include if similarity is above minimum threshold
            if best_product['similarity_score'] > 0.5:
                final_products.append({
                    'type': ptype,
                    'match_type': best_product['match_type'],
                    'matched_product_id': best_product['matched_product_id'],
                    'confidence': round(best_product['similarity_score'], 2)
                })
        
        return final_products
    
    def _empty_result(self, video_id: str) -> Dict[str, Any]:
        """Return empty result for failed processing"""
        return {
            "video_id": video_id,
            "vibes": [],
            "products": [],
            "metadata": {
                "error": "No frames could be extracted from video"
            }
        }
    
    def _error_result(self, video_id: str, error_msg: str) -> Dict[str, Any]:
        """Return error result"""
        return {
            "video_id": video_id,
            "vibes": [],
            "products": [],
            "metadata": {
                "error": error_msg
            }
        }
    
    def process_batch(self, video_paths: List[str], captions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple videos in batch
        
        Args:
            video_paths: List of video file paths
            captions: Optional list of captions (same length as video_paths)
            
        Returns:
            List of results for each video
        """
        results = []
        
        for i, video_path in enumerate(video_paths):
            caption = captions[i] if captions and i < len(captions) else None
            result = self.process_video(video_path, caption)
            results.append(result)
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_path}")
