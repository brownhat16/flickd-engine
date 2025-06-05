import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import numpy as np
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
from typing import List, Dict, Any, Optional
import logging

class ProductMatcher:
    def __init__(self, catalog_path: str, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize product matcher with CLIP model
        
        Args:
            catalog_path: Path to product catalog CSV
            model_name: CLIP model to use
        """
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Load and process catalog
        self.catalog_data = self._load_catalog(catalog_path)
        self.embeddings_index = self._build_embeddings_index()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_catalog(self, catalog_path: str) -> List[Dict[str, Any]]:
        """Load product catalog from CSV"""
        try:
            df = pd.read_csv(catalog_path)
            
            # Handle the provided catalog structure
            if 'image_url' in df.columns:
                # Use the images.csv format
                catalog = []
                for _, row in df.iterrows():
                    catalog.append({
                        'id': row['id'],
                        'image_url': row['image_url'],
                        'title': f"Product {row['id']}",  # Default title
                        'description': "",
                        'product_type': "unknown"
                    })
            else:
                # Use the product_data.csv format
                catalog = df.to_dict('records')
            
            self.logger.info(f"Loaded {len(catalog)} products from catalog")
            return catalog
            
        except Exception as e:
            self.logger.error(f"Error loading catalog: {e}")
            return []
    
    def _download_image(self, image_url: str) -> Optional[Image.Image]:
        """Download and return PIL Image from URL"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        except Exception as e:
            self.logger.warning(f"Failed to download image {image_url}: {e}")
            return None
    
    def _get_image_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for image"""
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize the embedding
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.cpu().numpy().flatten()
        except Exception as e:
            self.logger.error(f"Error getting image embedding: {e}")
            return np.zeros(512)  # Return zero embedding on error
    
    def _build_embeddings_index(self) -> faiss.IndexFlatIP:
        """Build FAISS index for product embeddings"""
        embeddings = []
        valid_products = []
        
        self.logger.info("Building embeddings index...")
        
        for i, product in enumerate(self.catalog_data):
            if i % 50 == 0:
                self.logger.info(f"Processing product {i}/{len(self.catalog_data)}")
            
            image = self._download_image(product['image_url'])
            if image:
                embedding = self._get_image_embedding(image)
                embeddings.append(embedding)
                valid_products.append(product)
            else:
                # Skip products with failed image downloads
                continue
        
        # Update catalog to only include products with valid embeddings
        self.catalog_data = valid_products
        
        if not embeddings:
            self.logger.error("No valid embeddings created!")
            # Create dummy index
            dummy_embedding = np.zeros((1, 512))
            index = faiss.IndexFlatIP(512)
            index.add(dummy_embedding.astype('float32'))
            return index
        
        embeddings = np.array(embeddings).astype('float32')
        
        # Build FAISS index for cosine similarity
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        self.logger.info(f"Built index with {len(embeddings)} embeddings")
        return index
    
    def match_detected_item(self, frame: np.ndarray, bbox: List[int], top_k: int = 5) -> Dict[str, Any]:
        """
        Match detected item with catalog products
        
        Args:
            frame: Input frame
            bbox: Bounding box [x, y, w, h]
            top_k: Number of top matches to return
            
        Returns:
            Dictionary with match results
        """
        try:
            # Crop detected item from frame
            x, y, w, h = bbox
            
            # Ensure coordinates are within frame bounds
            frame_h, frame_w = frame.shape[:2]
            x = max(0, min(x, frame_w - 1))
            y = max(0, min(y, frame_h - 1))
            w = max(1, min(w, frame_w - x))
            h = max(1, min(h, frame_h - y))
            
            cropped_item = frame[y:y+h, x:x+w]
            
            if cropped_item.size == 0:
                return self._no_match_result()
            
            cropped_image = Image.fromarray(cropped_item)
            
            # Get embedding for cropped item
            item_embedding = self._get_image_embedding(cropped_image)
            
            if np.allclose(item_embedding, 0):
                return self._no_match_result()
            
            item_embedding = item_embedding.reshape(1, -1).astype('float32')
            
            # Search in catalog
            similarities, indices = self.embeddings_index.search(item_embedding, k=min(top_k, len(self.catalog_data)))
            
            if len(indices[0]) == 0:
                return self._no_match_result()
            
            best_match_idx = indices[0][0]
            similarity_score = float(similarities[0][0])
            
            # Classify match quality
            if similarity_score > 0.9:
                match_type = "exact"
            elif similarity_score > 0.75:
                match_type = "similar"
            else:
                match_type = "no_match"
            
            return {
                'matched_product_id': self.catalog_data[best_match_idx]['id'],
                'similarity_score': similarity_score,
                'match_type': match_type,
                'product_info': self.catalog_data[best_match_idx],
                'all_matches': [
                    {
                        'product_id': self.catalog_data[indices[0][i]]['id'],
                        'similarity': float(similarities[0][i]),
                        'product_info': self.catalog_data[indices[0][i]]
                    }
                    for i in range(len(indices[0]))
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error in product matching: {e}")
            return self._no_match_result()
    
    def _no_match_result(self) -> Dict[str, Any]:
        """Return no match result"""
        return {
            'matched_product_id': None,
            'similarity_score': 0.0,
            'match_type': 'no_match',
            'product_info': None,
            'all_matches': []
        }
