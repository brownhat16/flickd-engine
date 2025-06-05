from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import json

from .video_processor import VideoProcessor
from .fashion_detector import FashionDetector
from .product_matcher import ProductMatcher
from .vibe_classifier import VibeClassifier

class FlickdEngine:
    def __init__(self, catalog_path: str, vibes_json_path: str = None):
        """
        Complete Flickd Smart Tagging & Vibe Classification Engine
        
        Implements the full hackathon requirements:
        1. Extract frames from videos
        2. Use YOLOv8 to identify fashion items  
        3. Match detected items to catalog using CLIP
        4. Analyze captions/transcripts to classify vibes using NLP
        5. Output structured data via JSON
        """
        # Initialize components
        self.video_processor = VideoProcessor(fps_extraction=2)
        self.fashion_detector = FashionDetector()
        self.product_matcher = ProductMatcher(catalog_path)
        self.vibe_classifier = VibeClassifier(vibes_json_path)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def process_video(self, video_path: str, caption: Optional[str] = None) -> Dict[str, Any]:
        """
        Main processing pipeline implementing hackathon requirements
        
        Args:
            video_path: Path to video file
            caption: Optional caption text with hashtags
            
        Returns:
            JSON output in hackathon-specified format
        """
        try:
            video_id = Path(video_path).stem
            self.logger.info(f"Processing video: {video_id}")
            
            # Step 1: Extract frames from videos
            self.logger.info("Extracting keyframes...")
            frames = self.video_processor.extract_keyframes(video_path)
            
            if not frames:
                return self._empty_result(video_id)
            
            # Step 2: Use YOLOv8 to identify fashion items
            self.logger.info("Detecting fashion items with YOLOv8...")
            all_detections = []
            
            for frame_data in frames:
                detections = self.fashion_detector.detect_fashion_items(
                    frame_data['frame'], 
                    frame_data['frame_number']
                )
                all_detections.extend(detections)
            
            self.logger.info(f"Found {len(all_detections)} fashion item detections")
            
            # Step 3: Match detected items to catalog using CLIP
            self.logger.info("Matching products with CLIP...")
            matched_products = []
            
            for detection in all_detections:
                frame_idx = detection['frame_number']
                if frame_idx < len(frames):
                    frame = frames[frame_idx]['frame']
                    
                    match_result = self.product_matcher.match_detected_item(
                        frame, detection['bbox']
                    )
                    
                    # Only include matches that meet hackathon criteria
                    if match_result['match_type'] != 'no_match':
                        product_info = {
                            'type': detection['class_name'],
                            'match_type': match_result['match_type'],
                            'matched_product_id': match_result['matched_product_id'],
                            'confidence': round(match_result['similarity_score'], 2)
                        }
                        matched_products.append(product_info)
            
            # Step 4: Analyze captions to classify vibes using NLP
            self.logger.info("Classifying vibes with NLP...")
            vibes = []
            if caption:
                vibes = self.vibe_classifier.classify_vibes(caption)
            
            # Step 5: Output structured data via JSON
            result = self._generate_hackathon_output(video_id, vibes, matched_products)
            
            self.logger.info(f"Processing complete for {video_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
            return self._error_result(Path(video_path).stem, str(e))
    
    def _generate_hackathon_output(self, video_id: str, vibes: List[str], products: List[Dict]) -> Dict[str, Any]:
        """
        Generate final JSON output according to exact hackathon specifications
        
        Expected format:
        {
            "video_id": "abc123",
            "vibes": ["Coquette", "Evening"],
            "products": [
                {
                    "type": "dress",
                    "color": "black", 
                    "match_type": "similar",
                    "matched_product_id": "prod_456",
                    "confidence": 0.84
                }
            ]
        }
        """
        
        # Aggregate and filter products by type
        aggregated_products = self._aggregate_products(products)
        
        # Build result in exact hackathon format
        result = {
            "video_id": video_id,
            "vibes": vibes,  # 1-3 vibes from official list
            "products": aggregated_products
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
        for product in products.items():
            ptype = product['type']
            if ptype not in type_groups:
                type_groups[ptype] = []
            type_groups[ptype].append(product)
        
        # Select best match for each type
        final_products = []
        
        for ptype, group in type_groups.items():
            # Sort by confidence and take best match
            best_product = max(group, key=lambda x: x['confidence'])
            
            # Add color extraction if possible (basic implementation)
            color = self._extract_color(best_product)
            
            final_products.append({
                'type': ptype,
                'color': color,
                'match_type': best_product['match_type'],
                'matched_product_id': best_product['matched_product_id'],
                'confidence': best_product['confidence']
            })
        
        return final_products
    
    def _extract_color(self, product: Dict) -> str:
        """
        Basic color extraction from product info
        In production, this would use computer vision color detection
        """
        # For now, return a default color - this would be enhanced with CV
        return "unknown"
    
    def _empty_result(self, video_id: str) -> Dict[str, Any]:
        """Return empty result for failed processing"""
        return {
            "video_id": video_id,
            "vibes": [],
            "products": []
        }
    
    def _error_result(self, video_id: str, error_msg: str) -> Dict[str, Any]:
        """Return error result"""
        return {
            "video_id": video_id,
            "vibes": [],
            "products": [],
            "error": error_msg
        }
