from ultralytics import YOLO
import torch
import numpy as np
from typing import List, Dict, Any

class FashionDetector:
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        """
        YOLOv8-based fashion item detector as specified in hackathon requirements
        
        Detects: tops, bottoms, dresses, jackets, accessories (earrings, bags, shoes)
        Returns: class name, bounding box (x, y, w, h), confidence score, frame number
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Fashion categories as per hackathon requirements
        self.fashion_categories = {
            'tops': ['shirt', 'blouse', 'top', 'sweater', 'hoodie', 't-shirt'],
            'bottoms': ['pants', 'jeans', 'trousers', 'leggings', 'shorts'],
            'dresses': ['dress', 'gown'],
            'jackets': ['jacket', 'coat', 'blazer', 'cardigan'],
            'shoes': ['shoes', 'boots', 'sneakers', 'sandals'],
            'bags': ['bag', 'handbag', 'backpack', 'purse'],
            'accessories': ['earrings', 'jewelry', 'watch', 'belt', 'hat', 'cap', 'sunglasses']
        }
    
    def detect_fashion_items(self, frame: np.ndarray, frame_number: int) -> List[Dict[str, Any]]:
        """
        Detect fashion items in frame using YOLOv8
        Returns detection results in hackathon-specified format
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
                    
                    # Filter for fashion items
                    fashion_category = self._get_fashion_category(class_name)
                    if fashion_category:
                        detection = {
                            'class_name': fashion_category,  # As required by hackathon
                            'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # (x, y, w, h)
                            'confidence_score': confidence,  # As required by hackathon
                            'frame_number': frame_number  # As required by hackathon
                        }
                        detections.append(detection)
        
        return detections
    
    def _get_fashion_category(self, class_name: str) -> str:
        """Map YOLO class to fashion category"""
        class_lower = class_name.lower()
        
        for category, keywords in self.fashion_categories.items():
            if any(keyword in class_lower for keyword in keywords):
                return category
        
        # Check if it's a person (we'll extract clothing from person detections)
        if 'person' in class_lower:
            return 'person'
        
        return None
