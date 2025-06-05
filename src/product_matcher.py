import torch
import clip
import faiss
import numpy as np
from PIL import Image
import pandas as pd
import requests
from io import BytesIO
from typing import List, Dict, Any, Optional
import logging

class ProductMatcher:
    def __init__(self, catalog_path: str):
        """
        CLIP + FAISS based product matcher as per hackathon requirements
        
        Uses cosine similarity to match detected items against catalog
        Labels results as: Exact Match (>0.9), Similar Match (0.75-0.9), No Match (<0.75)
        """
        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load catalog data
        self.catalog_data = self._load_catalog(catalog_path)
        self.embeddings_index = self._build_embeddings_index()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _load_catalog(self, catalog_path: str) -> List[Dict[str, Any]]:
        """Load product catalog from provided CSV files"""
        try:
            # Load images.csv for image URLs
            images_df = pd.read_csv('data/images.csv')
            
            # Load product_data.csv for product details
            products_df = pd.read_csv('data/product_data.csv')
            
            # Merge the data
            catalog = []
            for _, product in products_df.iterrows():
                product_images = images_df[images_df['id'] == product['id']]
                if not product_images.empty:
                    # Use first image for each product
                    image_url = product_images.iloc[0]['image_url']
                    
                    catalog.append({
                        'id': product['id'],
                        'title': product['title'],
                        'description': product['description'],
                        'product_type': product['product_type'],
                        'image_url': image_url,
                        'price': product.get('price_display_amount', 0)
                    })
            
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
    
    def _get_clip_embedding(self, image: Image.Image) -> np.ndarray:
        """Get CLIP embedding for image"""
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                # Normalize for cosine similarity
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                return image_features.cpu().numpy().flatten()
        except Exception as e:
            self.logger.error(f"Error getting CLIP embedding: {e}")
            return np.zeros(512)
    
    def _build_embeddings_index(self) -> faiss.IndexFlatIP:
        """Build FAISS index for product embeddings"""
        embeddings = []
        valid_products = []
        
        self.logger.info("Building CLIP embeddings index...")
        
        for i, product in enumerate(self.catalog_data):
            if i % 50 == 0:
                self.logger.info(f"Processing product {i}/{len(self.catalog_data)}")
            
            image = self._download_image(product['image_url'])
            if image:
                embedding = self._get_clip_embedding(image)
                if not np.allclose(embedding, 0):
                    embeddings.append(embedding)
                    valid_products.append(product)
        
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
        
        # Build FAISS index for cosine similarity (Inner Product with normalized vectors)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        self.logger.info(f"Built index with {len(embeddings)} embeddings")
        return index
    
    def match_detected_item(self, frame: np.ndarray, bbox: List[int]) -> Dict[str, Any]:
        """
        Match detected item with catalog products using CLIP + FAISS
        
        Returns match result with similarity classification as per hackathon specs:
        - Exact Match (similarity > 0.9)
        - Similar Match (0.75–0.9)  
        - No Match (< 0.75)
        """
        try:
            # Crop detected object from frame
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
            
            # Generate CLIP embedding for cropped item
            item_embedding = self._get_clip_embedding(cropped_image)
            
            if np.allclose(item_embedding, 0):
                return self._no_match_result()
            
            item_embedding = item_embedding.reshape(1, -1).astype('float32')
            
            # Search in catalog using FAISS
            similarities, indices = self.embeddings_index.search(item_embedding, k=1)
            
            if len(indices[0]) == 0:
                return self._no_match_result()
            
            best_match_idx = indices[0][0]
            similarity_score = float(similarities[0][0])
            
            # Classify match quality as per hackathon requirements
            if similarity_score > 0.9:
                match_type = "exact"
            elif similarity_score >= 0.75:
                match_type = "similar"
            else:
                match_type = "no_match"
            
            return {
                'matched_product_id': self.catalog_data[best_match_idx]['id'],
                'similarity_score': similarity_score,
                'match_type': match_type,
                'product_info': self.catalog_data[best_match_idx]
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
            'product_info': None
        }
