import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ==================== Novelty Constraint Layer ====================
class NoveltyConstraint(nn.Module):
    """Novelty Constraint Layer: Suppress repetitive generation, encourage problem innovation"""

    def __init__(self, similarity_threshold=0.85, penalty_factor=0.3):
        super().__init__()
        self.similarity_threshold = similarity_threshold
        self.penalty_factor = penalty_factor
        self.history_embeddings = []
        self.vectorizer = TfidfVectorizer(max_features=1000)

    def forward(self, x, current_question=""):
        """
        Args:
            x: Model output tensor
            current_question: Current generated question text
        """
        if current_question and len(self.history_embeddings) > 0:
            # Calculate similarity with historical questions
            current_embedding = self.vectorizer.transform([current_question])
            similarities = cosine_similarity(current_embedding, self.history_embeddings)
            max_similarity = np.max(similarities)

            if max_similarity > self.similarity_threshold:
                # Penalize repetitive questions
                return x * self.penalty_factor

        return x

    def update_history(self, question_text):
        """Update historical question database"""
        if hasattr(self.vectorizer, 'vocabulary_'):
            embedding = self.vectorizer.transform([question_text])
        else:
            # First use, need to fit
            if len(self.history_embeddings) == 0:
                self.vectorizer.fit([question_text])
            embedding = self.vectorizer.transform([question_text])

        self.history_embeddings.append(embedding)
