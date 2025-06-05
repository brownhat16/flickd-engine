from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import re

class VibeClassifier:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """
        Initialize vibe classifier
        
        Args:
            model_name: Sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        
        # Vibe definitions based on Flickd requirements
        self.vibe_definitions = {
            'Coquette': {
                'keywords': ['feminine', 'romantic', 'soft', 'pink', 'bow', 'lace', 'floral', 'delicate', 'sweet', 'girly', 'cute', 'pretty', 'dainty'],
                'description': 'Feminine, romantic style with soft colors, bows, lace, and delicate details. Sweet and girly aesthetic.'
            },
            'Clean Girl': {
                'keywords': ['minimal', 'natural', 'effortless', 'simple', 'fresh', 'dewy', 'neutral', 'understated', 'clean', 'basic'],
                'description': 'Minimalist, natural beauty with effortless styling and neutral tones. Simple and clean aesthetic.'
            },
            'Cottagecore': {
                'keywords': ['rustic', 'vintage', 'floral', 'pastoral', 'countryside', 'cozy', 'handmade', 'natural', 'cottage', 'rural', 'garden'],
                'description': 'Rustic, vintage-inspired aesthetic with floral patterns and countryside vibes. Cozy and natural.'
            },
            'Streetcore': {
                'keywords': ['urban', 'edgy', 'casual', 'street', 'hip-hop', 'sneakers', 'oversized', 'bold', 'cool', 'trendy', 'swag'],
                'description': 'Urban, edgy street style with casual and bold fashion choices. Cool and trendy vibe.'
            },
            'Y2K': {
                'keywords': ['futuristic', 'metallic', 'cyber', 'tech', 'holographic', 'neon', 'space-age', 'digital', 'retro-future', 'shiny'],
                'description': 'Futuristic, tech-inspired fashion with metallic and cyber elements. Retro-futuristic aesthetic.'
            },
            'Boho': {
                'keywords': ['bohemian', 'free-spirited', 'ethnic', 'flowing', 'earthy', 'layered', 'artistic', 'eclectic', 'hippie', 'relaxed'],
                'description': 'Bohemian, free-spirited style with flowing fabrics and earthy tones. Artistic and eclectic.'
            },
            'Party Glam': {
                'keywords': ['glamorous', 'sparkly', 'sequins', 'party', 'dressy', 'elegant', 'shiny', 'formal', 'glittery', 'fancy'],
                'description': 'Glamorous party wear with sparkles, sequins, and elegant details. Fancy and dressy.'
            }
        }
        
        # Pre-compute embeddings for vibe descriptions
        self.vibe_embeddings = self._compute_vibe_embeddings()
    
    def _compute_vibe_embeddings(self) -> Dict[str, np.ndarray]:
        """Pre-compute embeddings for each vibe"""
        embeddings = {}
        for vibe, info in self.vibe_definitions.items():
            # Combine keywords and description for richer representation
            text = f"{info['description']} {' '.join(info['keywords'])}"
            embedding = self.model.encode([text])
            embeddings[vibe] = embedding
        return embeddings
    
    def classify_vibes(self, text: str, top_k: int = 3, threshold: float = 0.3) -> List[str]:
        """
        Classify text into fashion vibes
        
        Args:
            text: Input text (caption, transcript, etc.)
            top_k: Maximum number of vibes to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of classified vibes
        """
        if not text or len(text.strip()) < 3:
            return []
        
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        if not text:
            return []
        
        # Get text embedding
        text_embedding = self.model.encode([text])
        
        # Calculate similarities with each vibe
        vibe_scores = {}
        for vibe, vibe_embedding in self.vibe_embeddings.items():
            similarity = cosine_similarity(text_embedding, vibe_embedding)[0][0]
            vibe_scores[vibe] = similarity
        
        # Sort by similarity and return top vibes
        sorted_vibes = sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Filter vibes with similarity > threshold
        relevant_vibes = [(vibe, score) for vibe, score in sorted_vibes if score > threshold]
        
        return [vibe for vibe, score in relevant_vibes[:top_k]]
    
    def classify_vibes_with_scores(self, text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Classify text into fashion vibes with confidence scores
        
        Returns:
            List of dictionaries with vibe and confidence score
        """
        if not text or len(text.strip()) < 3:
            return []
        
        # Clean and preprocess text
        text = self._preprocess_text(text)
        
        if not text:
            return []
        
        # Get text embedding
        text_embedding = self.model.encode([text])
        
        # Calculate similarities with each vibe
        vibe_scores = {}
        for vibe, vibe_embedding in self.vibe_embeddings.items():
            similarity = cosine_similarity(text_embedding, vibe_embedding)[0][0]
            vibe_scores[vibe] = similarity
        
        # Sort by similarity and return top vibes with scores
        sorted_vibes = sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'vibe': vibe, 'confidence': float(score)}
            for vibe, score in sorted_vibes[:top_k]
        ]
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove hashtags, mentions, and special characters
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower().strip()
    
    def analyze_keywords(self, text: str) -> Dict[str, List[str]]:
        """
        Analyze which keywords from each vibe are present in the text
        
        Returns:
            Dictionary mapping vibes to found keywords
        """
        text_lower = text.lower()
        found_keywords = {}
        
        for vibe, info in self.vibe_definitions.items():
            found = [keyword for keyword in info['keywords'] if keyword in text_lower]
            if found:
                found_keywords[vibe] = found
        
        return found_keywords
