from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import re
import json

class VibeClassifier:
    def __init__(self, vibes_json_path: str = None):
        """
        NLP-based vibe classifier for the 7 official Flickd vibes
        
        Classifies videos into 1-3 vibes from the official list:
        Coquette, Clean Girl, Cottagecore, Streetcore, Y2K, Boho, Party Glam
        """
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Official Flickd vibes from hackathon requirements
        self.official_vibes = [
            "Coquette",
            "Clean Girl", 
            "Cottagecore",
            "Streetcore",
            "Y2K",
            "Boho",
            "Party Glam"
        ]
        
        # Enhanced vibe definitions for better classification
        self.vibe_definitions = {
            'Coquette': {
                'keywords': ['feminine', 'romantic', 'soft', 'pink', 'bow', 'lace', 'floral', 'delicate', 'sweet', 'girly', 'cute', 'pretty', 'dainty', 'milkmaid', 'puff sleeves'],
                'description': 'Feminine, romantic style with soft colors, bows, lace, and delicate details. Sweet and girly aesthetic with puff sleeves and floral patterns.'
            },
            'Clean Girl': {
                'keywords': ['minimal', 'natural', 'effortless', 'simple', 'fresh', 'dewy', 'neutral', 'understated', 'clean', 'basic', 'no makeup', 'slicked back'],
                'description': 'Minimalist, natural beauty with effortless styling and neutral tones. Simple and clean aesthetic with no-makeup makeup look.'
            },
            'Cottagecore': {
                'keywords': ['rustic', 'vintage', 'floral', 'pastoral', 'countryside', 'cozy', 'handmade', 'natural', 'cottage', 'rural', 'garden', 'prairie', 'earthy'],
                'description': 'Rustic, vintage-inspired aesthetic with floral patterns and countryside vibes. Cozy and natural with handmade elements.'
            },
            'Streetcore': {
                'keywords': ['urban', 'edgy', 'casual', 'street', 'hip-hop', 'sneakers', 'oversized', 'bold', 'cool', 'trendy', 'swag', 'baggy', 'graphic'],
                'description': 'Urban, edgy street style with casual and bold fashion choices. Cool and trendy vibe with oversized fits and sneakers.'
            },
            'Y2K': {
                'keywords': ['futuristic', 'metallic', 'cyber', 'tech', 'holographic', 'neon', 'space-age', 'digital', 'retro-future', 'shiny', '2000s', 'chrome'],
                'description': 'Futuristic, tech-inspired fashion with metallic and cyber elements. Retro-futuristic aesthetic from the 2000s era.'
            },
            'Boho': {
                'keywords': ['bohemian', 'free-spirited', 'ethnic', 'flowing', 'earthy', 'layered', 'artistic', 'eclectic', 'hippie', 'relaxed', 'fringe', 'paisley'],
                'description': 'Bohemian, free-spirited style with flowing fabrics and earthy tones. Artistic and eclectic with layered accessories.'
            },
            'Party Glam': {
                'keywords': ['glamorous', 'sparkly', 'sequins', 'party', 'dressy', 'elegant', 'shiny', 'formal', 'glittery', 'fancy', 'cocktail', 'evening'],
                'description': 'Glamorous party wear with sparkles, sequins, and elegant details. Fancy and dressy for special occasions.'
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
    
    def classify_vibes(self, text: str, max_vibes: int = 3) -> List[str]:
        """
        Classify text into 1-3 fashion vibes from the official Flickd list
        
        Args:
            text: Input text (caption + hashtags or audio transcript)
            max_vibes: Maximum number of vibes to return (1-3 as per hackathon)
            
        Returns:
            List of classified vibes (1-3 vibes max)
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
        
        # Apply threshold and return top vibes (1-3 max)
        threshold = 0.25  # Lower threshold for better recall
        relevant_vibes = [(vibe, score) for vibe, score in sorted_vibes if score > threshold]
        
        # Return top vibes (max 3 as per hackathon requirements)
        return [vibe for vibe, score in relevant_vibes[:max_vibes]]
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove hashtags but keep the content
        text = re.sub(r'#(\w+)', r'\1', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove extra whitespace and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        return text.lower().strip()
