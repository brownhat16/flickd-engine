from ultralytics import YOLO
import torch
import numpy as np
from typing import List, Dict, Any

class FashionDetector:
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        Initialize fashion detector with YOLOv8
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Fashion categories mapping based on YOLO classes
        self.fashion_categories = {
            'person': ['person'],
            'shirt': ['shirt', 'blouse', 'top', 'sweater', 'hoodie'],
            'pants': ['pants', 'jeans', 'trousers', 'leggings'],
            'dress': ['dress', 'gown'],
            'jacket': ['jacket', 'coat', 'blazer', 'cardigan'],
            'shoes': ['shoes', 'boots', 'sneakers', 'sandals'],
            'bag': ['bag', 'handbag', 'backpack', 'purse'],
            'accessories': ['hat', 'cap', 'sunglasses', 'jewelry', 'watch', 'belt']
        }
        
        # YOLO class names that are fashion-related
        self.fashion_keywords = [
            'person', 'shirt', 'pants', 'dress', 'jacket', 'shoes', 'bag', 
            'handbag', 'backpack', 'hat', 'cap', 'sunglasses'
        ]
    
    def detect_fashion_items(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect fashion items in frame using YOLOv8
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detection dictionaries
        """
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    # Filter for fashion items or person (to extract clothing from)
                    if self._is_fashion_item(class_name):
                        fashion_category = self._map_to_fashion_category(class_name)
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # [x, y, w, h]
                            'confidence': confidence,
                            'class': fashion_category,
                            'raw_class': class_name,
                            'class_id': class_id
                        }
                        detections.append(detection)
        
        return detections
    
    def _is_fashion_item(self, class_name: str) -> bool:
        """Check if detected class is fashion-related"""
        return any(keyword in class_name.lower() for keyword in self.fashion_keywords)
    
    def _map_to_fashion_category(self, class_name: str) -> str:
        """Map YOLO class to fashion category"""
        class_lower = class_name.lower()
        
        for category, keywords in self.fashion_categories.items():
            if any(keyword in class_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def extract_person_clothing(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract clothing items from detected persons
        This is a simplified approach - in production, you'd use a specialized model
        """
        detections = self.detect_fashion_items(frame)
        
        # Find person detections
        person_detections = [d for d in detections if d['class'] == 'person']
        clothing_detections = []
        
        for person in person_detections:
            x, y, w, h = person['bbox']
            
            # Estimate clothing regions within person bbox
            # Top region (shirt/dress)
            top_region = {
                'bbox': [x, y, w, int(h * 0.6)],
                'confidence': person['confidence'] * 0.8,  # Lower confidence for estimated regions
                'class': 'shirt',
                'raw_class': 'estimated_shirt',
                'estimated': True
            }
            
            # Bottom region (pants/skirt)
            bottom_region = {
                'bbox': [x, y + int(h * 0.5), w, int(h * 0.5)],
                'confidence': person['confidence'] * 0.7,
                'class': 'pants',
                'raw_class': 'estimated_pants',
                'estimated': True
            }
            
            clothing_detections.extend([top_region, bottom_region])
        
        # Combine with direct fashion detections
        all_detections = [d for d in detections if d['class'] != 'person'] + clothing_detections
        
        return all_detections
