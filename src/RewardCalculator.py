
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from src.config import TrainingConfig

# ==================== Reward Calculator ====================
class RewardCalculator:
    """Multi-dimensional reward function calculator"""

    def __init__(self):
        self.difficulty_keywords = {
            1: ['basic', 'simple', 'elementary'],
            2: ['intermediate', 'moderate'],
            3: ['advanced', 'complex'],
            4: ['challenging', 'difficult'],
            5: ['expert', 'olympiad', 'extremely']
        }

        self.rigor_keywords = ['prove', 'theorem', 'lemma', 'contradiction', 'induction',
                              'necessary', 'sufficient', 'if and only if', 'analysis']

    def calculate_difficulty(self, question: str) -> float:
        """Calculate problem difficulty (0-1)"""
        difficulty_score = 0
        text_lower = question.lower()

        # Keyword-based scoring
        for level, keywords in self.difficulty_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    difficulty_score = max(difficulty_score, level / 5.0)

        # Text complexity scoring
        complexity_score = min(len(question) / 500.0, 1.0)  # Text length
        math_symbols = len(re.findall(r'[∑∏∫∂∇≤≥≠±∞]', question)) / 10.0

        return min((difficulty_score + complexity_score + math_symbols) / 3.0, 1.0)

    def calculate_novelty(self, question: str, history_questions: List[str]) -> float:
        """Calculate problem novelty (0-1)"""
        if not history_questions:
            return 1.0

        # Use TF-IDF for similarity calculation
        vectorizer = TfidfVectorizer(max_features=100)
        all_questions = history_questions + [question]

        try:
            tfidf_matrix = vectorizer.fit_transform(all_questions)
            similarities = cosine_similarity(tfidf_matrix[-1:], tfidf_matrix[:-1])
            max_similarity = np.max(similarities)
            return 1.0 - max_similarity
        except:
            return 0.5  # Default value if calculation fails

    def calculate_rigor(self, question: str) -> float:
        """Calculate logical rigor (0-1)"""
        text_lower = question.lower()
        rigor_count = sum(1 for keyword in self.rigor_keywords if keyword in text_lower)

        # Logical structure scoring
        proof_indicators = len(re.findall(r'(prove|show|demonstrate|verify)', text_lower))
        logical_connectors = len(re.findall(r'(therefore|thus|hence|because|since)', text_lower))

        rigor_score = (rigor_count + proof_indicators + logical_connectors) / 10.0
        return min(rigor_score, 1.0)

    def calculate_diversity(self, question: str, group_questions: List[str]) -> float:
        """Calculate group diversity (0-1)"""
        if len(group_questions) <= 1:
            return 1.0

        # Calculate average similarity within the group
        try:
            vectorizer = TfidfVectorizer(max_features=50)
            tfidf_matrix = vectorizer.fit_transform(group_questions + [question])
            similarities = cosine_similarity(tfidf_matrix)

            # Calculate average similarity (excluding self-similarity)
            n = len(similarities)
            total_similarity = np.sum(similarities) - np.trace(similarities)
            avg_similarity = total_similarity / (n * (n - 1))

            return 1.0 - avg_similarity
        except:
            return 0.5

    def calculate_reward(self, question: str, history_questions: List[str],
                        group_questions: List[str]) -> float:
        """Calculate comprehensive reward"""
        difficulty = self.calculate_difficulty(question)
        novelty = self.calculate_novelty(question, history_questions)
        rigor = self.calculate_rigor(question)
        diversity = self.calculate_diversity(question, group_questions)

        # Weighted combination
        weights = TrainingConfig.REWARD_WEIGHTS
        reward = (weights['difficulty'] * difficulty +
                 weights['novelty'] * novelty +
                 weights['rigor'] * rigor +
                 weights['diversity'] * diversity)

        return reward

# ==================== DeepSeek-R1 Teacher Model
    def calculate_normalized_reward(self, question: str, history_questions: List[str],
                                  group_questions: List[str]) -> float:
        """计算标准化奖励"""
        # 计算原始奖励
        raw_reward = self.calculate_reward(question, history_questions, group_questions)

        # 如果有历史奖励，进行标准化
        if hasattr(self, 'reward_history') and self.reward_history:
            mean_reward = np.mean(self.reward_history)
            std_reward = np.std(self.reward_history) + 1e-8  # 避免除零
            normalized_reward = (raw_reward - mean_reward) / std_reward
        else:
            normalized_reward = raw_reward

        # 更新奖励历史
        if not hasattr(self, 'reward_history'):
            self.reward_history = []
        self.reward_history.append(raw_reward)

        # 保持最近1000个奖励
        if len(self.reward_history) > 1000:
            self.reward_history = self.reward_history[-1000:]

        return normalized_reward
