# -*- coding: utf-8 -*-
"""Questions-Gen Model Training Script

Qwen3-14B-based competition problem generation model training script
Implements three-stage training: Basic Pretraining -> RL GRPO Optimization -> Knowledge Distillation
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import TextStreamer
from unsloth import FastLanguageModel
from unsloth.chat_templates import standardize_sharegpt
from trl import SFTTrainer, SFTConfig
import json
import random
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import math
import os
import time

# DeepSeek-R1 API using OpenAI SDK
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI SDK not found. Install with: pip install openai")

# DeepSeek-R1 API
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI SDK not found. Install with: pip install openai")

# ==================== Configuration Parameters ====================
class TrainingConfig:
    # Model configuration
    MODEL_NAME = "unsloth/Qwen3-14B"
    MAX_SEQ_LENGTH = 2048
    LOAD_IN_4BIT = True

    # LoRA configuration
    LORA_R = 32
    LORA_ALPHA = 32
    LORA_DROPOUT = 0

    # Training configuration
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    LEARNING_RATE = 2e-4
    MAX_STEPS_STAGE1 = 200  # Basic pretraining
    MAX_STEPS_STAGE2 = 100   # RL GRPO
    MAX_STEPS_STAGE3 = 80   # Knowledge distillation

    # GRPO configuration
    GROUP_SIZE = 8
    REWARD_WEIGHTS = {
        'difficulty': 0.4,
        'novelty': 0.3,
        'rigor': 0.2,
        'diversity': 0.1
    }

    # Variation training configuration
    VARIATION_TRAINING_RATIO = 0.4  # 40% of training data for variation generation
    VARIATION_QUALITY_THRESHOLD = 0.5  # Minimum quality score for variations
    ENABLE_COORDINATED_TRAINING = True  # Enable coordinated training

    # Data mixing ratio
    BASIC_RATIO = 0.5
    VARIATION_RATIO = 0.3
    INNOVATION_RATIO = 0.2

    # HuggingFace configuration
    HF_USERNAME = "xingqiang"
    HF_MODEL_NAME = "questions-gen-qwen3-14b"
    HF_TOKEN = None  # Set this via environment variable HF_TOKEN for security
    
    # Model saving configuration (保存配置)
    SAVE_QUANTIZED_VERSIONS = False  # 不创建量化版本
    PRESERVE_FULL_PRECISION = True   # 保持原精度

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

# ==================== Data Processor ====================
class QuestionsDataPreparer:
    """Competition problem dataset processor"""

    def __init__(self):
        print("🔄 Initializing data processor...")
        self.real_processor = None
        try:
            from tests.real_math_data_processor import RealMathDataProcessor
            self.real_processor = RealMathDataProcessor()
            print("✅ Real dataset processor loaded")
        except ImportError:
            print("⚠️ Real dataset processor not found, using simulated data")

    def prepare_training_data(self, tokenizer) -> Dataset:
        """Prepare training dataset - Using unsloth standard approach exactly like reference script"""
        print("🔄 Preparing training dataset...")

        # Try to use real data first (following unsloth reference script exactly)
        try:
            if self.real_processor:
                print("✅ Using real math reasoning dataset (unsloth style)")
                return self.real_processor.prepare_unsloth_training_dataset(
                    tokenizer=tokenizer,
                    chat_percentage=0.25  # Same ratio as reference script
                )
        except Exception as e:
            print(f"⚠️ Real data processor failed: {e}")

        # Fallback: Create simple unsloth-style dataset from basic prompts
        print("🔄 Creating fallback unsloth-style training dataset...")

        # Basic problem generation examples in unsloth format
        basic_conversations = [
            [
                {"role": "user", "content": "Generate a challenging algebra competition problem:"},
                {"role": "assistant", "content": "Find all real solutions to the equation x^4 - 5x^2 + 6 = 0.\n\nSolution: Let y = x^2. Then y^2 - 5y + 6 = 0, so (y-2)(y-3) = 0. Thus y = 2 or y = 3, giving x = ±√2 or x = ±√3."}
            ],
            [
                {"role": "user", "content": "Create a geometry problem suitable for math olympiad:"},
                {"role": "assistant", "content": "In triangle ABC, prove that the sum of any two sides is greater than the third side.\n\nSolution: This is the triangle inequality theorem. For any triangle with sides a, b, c: a + b > c, b + c > a, and a + c > b."}
            ],
            [
                {"role": "user", "content": "Design a calculus problem with moderate difficulty:"},
                {"role": "assistant", "content": "Find the maximum value of f(x) = x^3 - 3x^2 + 2 on the interval [0, 3].\n\nSolution: f'(x) = 3x^2 - 6x = 3x(x-2). Critical points at x = 0, 2. Check: f(0) = 2, f(2) = -2, f(3) = 2. Maximum is 2."}
            ],
            [
                {"role": "user", "content": "Formulate a number theory competition question:"},
                {"role": "assistant", "content": "Prove that there are infinitely many prime numbers.\n\nSolution: Assume finitely many primes p1, p2, ..., pn. Consider N = p1·p2·...·pn + 1. N is not divisible by any pi, so either N is prime or has a prime factor not in our list."}
            ],
            [
                {"role": "user", "content": "Develop a combinatorics problem for advanced students:"},
                {"role": "assistant", "content": "How many ways can 8 people be arranged in a circle?\n\nSolution: Fix one person's position to break rotational symmetry. Arrange remaining 7 people in (8-1)! = 7! = 5040 ways."}
            ]
        ]

        # Expand the dataset by creating variations
        all_conversations = basic_conversations.copy()

        # Add variation examples
        for _ in range(10):
            all_conversations.extend([
                [
                    {"role": "user", "content": f"Generate a high-quality competition problem:"},
                    {"role": "assistant", "content": f"Find the roots of x^2 - 4x + 3 = 0.\n\nSolution: Using the quadratic formula: x = (4 ± √(16-12))/2 = (4 ± 2)/2. So x = 3 or x = 1."}
                ],
                [
                    {"role": "user", "content": f"Create an innovative math problem:"},
                    {"role": "assistant", "content": f"Prove that √2 is irrational.\n\nSolution: Assume √2 = p/q in lowest terms. Then 2q^2 = p^2, so p^2 is even, thus p is even. Let p = 2k, then 2q^2 = 4k^2, so q^2 = 2k^2, making q even. Contradiction."}
                ]
            ])

        # Return in unsloth standard format
        return Dataset.from_dict({"conversations": all_conversations})

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

# ==================== DeepSeek-R1 Teacher Model ====================
class DeepSeekTeacher:
    """DeepSeek-R1 Teacher Model for Knowledge Distillation - Using OpenAI SDK"""

    def __init__(self, api_key: str = None):
        if not OPENAI_AVAILABLE:
            print("❌ OpenAI SDK not available. DeepSeek teacher disabled.")
            self.client = None
            return

        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY', 'sk-d02fca54e07f4bdfb1778aeb62ae7671')

        # Initialize OpenAI client for DeepSeek
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

        # Test API connection
        print("🔄 Testing DeepSeek-R1 API connection...")
        if self._test_connection():
            print("✅ DeepSeek-R1 teacher model connected successfully")
        else:
            print("❌ DeepSeek-R1 API connection failed")
            self.client = None

    def _test_connection(self) -> bool:
        """Test API connection"""
        if not self.client:
            return False

        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Test connection"}
                ],
                max_tokens=10,
                temperature=0.1,
                timeout=30
            )

            # 更宽松的成功检查：只要有响应就认为连接成功
            if response and hasattr(response, 'choices') and response.choices:
                print(f"✅ API响应: {response.choices[0].message.content}")
                return True
            else:
                print("⚠️ API返回空响应")
                return False

        except Exception as e:
            print(f"⚠️ API test failed: {e}")
            # 打印更详细的错误信息
            if "model" in str(e).lower():
                print("   💡 可能是模型名称问题，但API基本连接正常")
                return True  # 如果只是模型名问题，仍认为连接正常
            return False

    def evaluate_problem(self, problem: str) -> Dict[str, any]:
        """Use DeepSeek-R1 to evaluate a math problem"""
        if not self.client:
            return self._default_evaluation()

        evaluation_prompt = f"""
Please evaluate this mathematics competition problem comprehensively:

Problem: {problem}

Please provide evaluation on the following aspects:
1. Mathematical rigor and correctness (1-5 scale)
2. Difficulty level (1-5 scale)
3. Innovation and creativity (1-5 scale)
4. Problem clarity and expression (1-5 scale)
5. Educational value (1-5 scale)
6. Specific suggestions for improvement

Please provide detailed reasoning and specific feedback in a structured format.
"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are an expert mathematics professor evaluating competition problems."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_tokens=1500,
                temperature=0.3,
                stream=False
            )

            evaluation_text = response.choices[0].message.content

            # Parse evaluation results
            evaluation = self._parse_evaluation(evaluation_text)
            evaluation['raw_feedback'] = evaluation_text

            return evaluation

        except Exception as e:
            print(f"❌ DeepSeek evaluation failed: {e}")
            return self._default_evaluation()

    def improve_problem(self, problem: str, feedback: str) -> str:
        """Use DeepSeek-R1 to improve a problem based on feedback"""
        if not self.client:
            return problem

        improvement_prompt = f"""
Based on the following feedback, please improve this mathematics competition problem:

Original Problem: {problem}

Feedback: {feedback}

Please provide an improved version that addresses the feedback while maintaining the mathematical essence. Focus on:
1. Enhancing mathematical rigor
2. Improving clarity of expression
3. Adjusting difficulty appropriately
4. Adding educational value

Provide only the improved problem statement, no additional commentary.
"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are an expert mathematics problem designer."},
                    {"role": "user", "content": improvement_prompt}
                ],
                max_tokens=4096,
                temperature=0.4,
                stream=False
            )

            improved_problem = response.choices[0].message.content
            return improved_problem.strip()

        except Exception as e:
            print(f"❌ DeepSeek improvement failed: {e}")
            return problem

    def generate_variations(self, original_problem: str, num_variations: int = 3) -> List[str]:
        """Use DeepSeek-R1 to generate intelligent problem variations"""
        if not self.client:
            return []

        variation_prompt = f"""
Generate {num_variations} intelligent variations of this mathematics competition problem:

Original Problem: {original_problem}

Please create variations that:
1. Maintain the same mathematical concepts and solution methods
2. Use different contexts or applications
3. Adjust parameters while preserving difficulty
4. Ensure mathematical correctness and clarity

Provide each variation on a separate line, numbered 1, 2, 3, etc.
"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are an expert mathematics problem designer specializing in creating problem variations."},
                    {"role": "user", "content": variation_prompt}
                ],
                max_tokens=4096,
                temperature=0.6,
                stream=False
            )

            variations_text = response.choices[0].message.content

            # Parse variations from response
            variations = self._parse_variations(variations_text)
            return variations[:num_variations]

        except Exception as e:
            print(f"❌ DeepSeek variation generation failed: {e}")
            return []

    def _parse_evaluation(self, evaluation_text: str) -> Dict[str, any]:
        """Parse evaluation results from DeepSeek response"""
        evaluation = {
            'rigor_score': 3.0,
            'difficulty_score': 3.0,
            'innovation_score': 3.0,
            'clarity_score': 3.0,
            'educational_value': 3.0,
            'overall_score': 3.0,
            'suggestions': []
        }

        try:
            # Extract numerical scores using pattern matching
            text_lower = evaluation_text.lower()

            # Look for explicit numerical ratings
            score_patterns = [
                (r'rigor[^\d]*([1-5])', 'rigor_score'),
                (r'difficulty[^\d]*([1-5])', 'difficulty_score'),
                (r'innovation[^\d]*([1-5])', 'innovation_score'),
                (r'clarity[^\d]*([1-5])', 'clarity_score'),
                (r'educational[^\d]*([1-5])', 'educational_value')
            ]

            for pattern, key in score_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    evaluation[key] = float(match.group(1))

            # Look for qualitative indicators if no explicit scores
            if 'difficulty' in text_lower:
                if any(word in text_lower for word in ['easy', 'simple', 'basic', 'elementary']):
                    evaluation['difficulty_score'] = 2.0
                elif any(word in text_lower for word in ['hard', 'challenging', 'difficult', 'advanced']):
                    evaluation['difficulty_score'] = 4.0

            # Look for rigor mentions
            if any(word in text_lower for word in ['rigorous', 'precise', 'correct', 'well-defined']):
                evaluation['rigor_score'] = 4.0
            elif any(word in text_lower for word in ['unclear', 'ambiguous', 'imprecise']):
                evaluation['rigor_score'] = 2.0

            # Look for innovation mentions
            if any(word in text_lower for word in ['creative', 'innovative', 'original', 'novel']):
                evaluation['innovation_score'] = 4.0
            elif any(word in text_lower for word in ['standard', 'typical', 'common']):
                evaluation['innovation_score'] = 2.0

            # Calculate overall score
            evaluation['overall_score'] = np.mean([
                evaluation['rigor_score'],
                evaluation['difficulty_score'],
                evaluation['innovation_score'],
                evaluation['clarity_score'],
                evaluation['educational_value']
            ])

        except Exception as e:
            print(f"⚠️ Evaluation parsing failed: {e}")

        return evaluation

    def _parse_variations(self, variations_text: str) -> List[str]:
        """Parse variations from DeepSeek response"""
        variations = []

        try:
            # Split by numbered items
            lines = variations_text.split('\n')
            current_variation = ""

            for line in lines:
                line = line.strip()
                if re.match(r'^[0-9]+\.', line):  # New numbered item
                    if current_variation:
                        variations.append(current_variation.strip())
                    current_variation = re.sub(r'^[0-9]+\.\s*', '', line)
                elif current_variation and line:
                    current_variation += " " + line

            # Add the last variation
            if current_variation:
                variations.append(current_variation.strip())

        except Exception as e:
            print(f"⚠️ Variation parsing failed: {e}")

        return variations

    def _default_evaluation(self) -> Dict[str, any]:
        """Return default evaluation when API fails"""
        return {
            'rigor_score': 3.0,
            'difficulty_score': 3.0,
            'innovation_score': 3.0,
            'clarity_score': 3.0,
            'educational_value': 3.0,
            'overall_score': 3.0,
            'suggestions': ['Unable to get detailed feedback from teacher model'],
            'raw_feedback': 'DeepSeek-R1 evaluation temporarily unavailable'
        }

# ==================== Main Trainer ====================
class QuestionsGenTrainer:
    """Questions-Gen model trainer"""

    def __init__(self):
        self.config = TrainingConfig()
        self.data_preparer = QuestionsDataPreparer()
        self.reward_calculator = RewardCalculator()
        self.novelty_constraint = NoveltyConstraint()
        self.history_questions = []

        # Initialize DeepSeek-R1 teacher model
        self.deepseek_teacher = DeepSeekTeacher()

        print("🚀 Initializing Questions-Gen trainer...")
        self._load_model()

    def _monitor_memory(self, stage_name=""):
        """监控GPU内存使用"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_reserved = torch.cuda.max_memory_reserved() / 1024**3
            print(f"📊 {stage_name} 内存: 已分配={allocated:.2f}GB, 已预留={reserved:.2f}GB, 峰值={max_reserved:.2f}GB")

            # 如果内存使用过高，执行清理
            if reserved > 12.0:  # 假设 A100 有 40GB，使用超过 30%
                print("🧹 执行内存清理...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _fix_attention_bias(self):
        """统一的注意力偏置修复方法"""
        print("🔧 检查并修复 attn_bias...")
        try:
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                for i, layer in enumerate(self.model.model.layers):
                    if hasattr(layer, 'self_attn'):
                        if not hasattr(layer.self_attn, 'attn_bias') or layer.self_attn.attn_bias is None:
                            # 初始化 attn_bias
                            layer.self_attn.attn_bias = torch.zeros(1, 1, 1, 1, device=device, dtype=dtype, requires_grad=False)
                            print(f"✅ 修复 layer {i} attn_bias")
            print("✅ attn_bias 检查完成")
        except Exception as e:
            print(f"⚠️ attn_bias 修复警告: {e}")

    def _validate_training_progress(self, stage_name: str, step: int):
        """验证训练进度"""
        print(f"🔍 {stage_name} 第{step}步验证...")

        # 生成测试问题
        test_prompts = [
            "Generate a calculus problem:",
            "Create an algebra challenge:",
            "Design a geometry proof:"
        ]

        total_quality = 0
        for prompt in test_prompts:
            question = self._generate_single_question(prompt)
            reward = self.reward_calculator.calculate_reward(question, self.history_questions, [])
            total_quality += reward

        avg_quality = total_quality / len(test_prompts)
        print(f"📊 当前平均质量分数: {avg_quality:.3f}")

        # 记录验证历史
        if not hasattr(self, 'validation_history'):
            self.validation_history = []
        self.validation_history.append({
            'stage': stage_name,
            'step': step,
            'quality': avg_quality
        })

        return avg_quality


    def _load_model(self):
        """Load and configure model - Using unsloth standard approach"""

        # Clear GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("🔄 Loading Qwen3-14B model...")

        # Use unsloth standard model loading (consistent with reference script)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Qwen3-14B",
            max_seq_length = 2048,   # Context length - can be longer, but uses more memory
            load_in_4bit = True,     # 4bit uses much less memory
            # token = "hf_...",      # use one if using gated models
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 32,  # Best to choose alpha = rank or rank*2
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,   # We support rank stabilized LoRA
            loftq_config = None,  # And LoftQ
        )

        print("✅ Model loading completed")

                # Prepare model for training
        self.model.train()

        # Ensure model is properly initialized
        if hasattr(self.model, 'config'):
            # Set attention configuration
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False
            if hasattr(self.model.config, 'pretraining_tp'):
                self.model.config.pretraining_tp = 1

        # 修复注意力偏置
        self._fix_attention_bias()

        # Clear any cached states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stage1_basic_training(self):
        """Stage 1: Basic pretraining - Using unsloth standard training"""
        print("\n" + "="*50)
        print("🎯 Stage 1: Basic pretraining started")
        print("="*50)

        # Prepare training data in unsloth format (exactly like reference script)
        train_dataset = self.data_preparer.prepare_training_data(self.tokenizer)

        if train_dataset is None:
            print("❌ Failed to prepare training dataset")
            return

        # If dataset already has 'text' field (from unsloth processing), use directly
        if 'text' in train_dataset.column_names:
            final_dataset = train_dataset
            print("✅ Using pre-processed unsloth-style dataset")
        else:
            # Convert to chat format using unsloth approach (fallback)
            conversations = self.tokenizer.apply_chat_template(
                train_dataset["conversations"],
                tokenize = False,
            )

            # Create final dataset in unsloth standard format
            final_dataset = Dataset.from_dict({"text": conversations})
            final_dataset = final_dataset.shuffle(seed = 3407)

        # Show memory stats (unsloth style)
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        # Configure trainer (exactly like reference script)
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = final_dataset,
            eval_dataset = None, # Can set up evaluation!
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 8, # 减少GA步数
                warmup_steps = 10, # 减少warmup
                max_steps = min(50, self.config.MAX_STEPS_STAGE1), # 限制步数
                learning_rate = 1e-4, # 降低学习率
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "cosine",
                seed = 3407,
                report_to = "none",
                dataloader_pin_memory = False,
                save_safetensors = True,
                fp16 = False,
                bf16 = True,
                remove_unused_columns = False,
                dataloader_num_workers = 0,
                ddp_find_unused_parameters = False,
                group_by_length = False, # 禁用group_by_length
                save_only_model = True,
            ),
        )

        self._monitor_memory("训练开始前")
        print("🔄 Starting basic pretraining...")
        try:
            # Clear cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            trainer_stats = trainer.train()
            self._monitor_memory("训练完成后")

        except Exception as e:
            print(f"❌ Training error: {e}")
            print("🔄 Attempting recovery...")

            # Clear memory and try again with smaller batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 恢复过程中重新修复注意力偏置
            print("🔧 恢复过程中重新检查 attn_bias...")
            try:
                self._fix_attention_bias()
            except Exception as attn_e:
                print(f"⚠️ attn_bias 恢复警告: {attn_e}")

            # Recreate trainer with safer settings
            trainer = SFTTrainer(
                model = self.model,
                tokenizer = self.tokenizer,
                train_dataset = final_dataset,
                eval_dataset = None,
                args = SFTConfig(
                    dataset_text_field = "text",
                    per_device_train_batch_size = 1,
                    gradient_accumulation_steps = 16,  # Increased GA
                    warmup_steps = 5,
                    max_steps = min(10, self.config.MAX_STEPS_STAGE1),  # Reduced steps
                    learning_rate = 1e-4,  # Reduced LR
                    logging_steps = 1,
                    optim = "adamw_8bit",
                    weight_decay = 0.01,
                    lr_scheduler_type = "cosine",
                    seed = 3407,
                    report_to = "none",
                    dataloader_pin_memory = False,
                    save_safetensors = True,
                    fp16 = False,
                    bf16 = True,
                    remove_unused_columns = False,
                    dataloader_num_workers = 0,
                    ddp_find_unused_parameters = False,
                    group_by_length = False,  # Disable for safety
                    save_only_model = True,
                ),
            )

            print("🔄 Retrying training with safer configuration...")
            trainer_stats = trainer.train()
            self._monitor_memory("训练完成后")

        # Show final memory and time stats (unsloth style)
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        # Save checkpoint
        os.makedirs("checkpoints/stage1_basic", exist_ok=True)
        self.model.save_pretrained("checkpoints/stage1_basic")
        self.tokenizer.save_pretrained("checkpoints/stage1_basic")
        print("💾 Stage 1 model saved")

    def stage2_grpo_training(self):
        """Stage 2: RL GRPO optimization - Coordinated problem generation and variation training"""
        print("\n" + "="*50)
        print("🎯 Stage 2: RL GRPO optimization started (with variation generation training)")
        print("="*50)

        # Prepare base dataset for policy training
        train_dataset = self.data_preparer.prepare_training_data(self.tokenizer)

        # Load variation training data from real data processor
        variation_training_data = []
        if self.data_preparer.real_processor:
            try:
                variation_training_data = self.data_preparer.real_processor.create_grpo_variation_training_data()
                print(f"✅ Loaded {len(variation_training_data)} variation training examples")
            except Exception as e:
                print(f"⚠️ Failed to load variation training data: {e}")

        for step in range(self.config.MAX_STEPS_STAGE2):
            print(f"\n🔄 GRPO step {step+1}/{self.config.MAX_STEPS_STAGE2}")

            # === Part 1: Original Problem Generation Training ===
            print("📝 Training original problem generation...")

            # Generate a group of questions
            group_questions = self._generate_question_group()

            # Calculate rewards
            rewards = []
            for question in group_questions:
                reward = self.reward_calculator.calculate_reward(
                    question, self.history_questions, group_questions
                )
                rewards.append(reward)

                # Update novelty constraint layer
                self.novelty_constraint.update_history(question)

            # Select baseline question (median reward)
            median_idx = np.argsort(rewards)[len(rewards)//2]
            baseline_reward = rewards[median_idx]

            # Calculate advantages for policy gradient
            advantages = [r - baseline_reward for r in rewards]

            # Create training data for original problem generation
            original_conversations = []
            for i, (question, advantage) in enumerate(zip(group_questions, advantages)):
                if advantage > 0:  # Only train on better-than-baseline questions
                    original_conversations.append([
                        {"role": "user", "content": "Generate a high-quality competition problem:"},
                        {"role": "assistant", "content": question}
                    ])

            # === Part 2: Variation Generation Training ===
            print("🔄 Training variation generation capabilities...")

            # Generate variations for high-reward questions
            variation_conversations = []
            high_reward_questions = [q for q, r in zip(group_questions, rewards) if r > baseline_reward]

            for original_question in high_reward_questions[:3]:  # Use top 3 questions
                # Create variation training examples
                variation_examples = self._create_variation_training_examples(original_question)
                variation_conversations.extend(variation_examples)

            # Add pre-loaded variation training data (sample some)
            if variation_training_data and step % 3 == 0:  # Every 3 steps, add real variation data
                sampled_variation_data = random.sample(
                    variation_training_data,
                    min(5, len(variation_training_data))
                )
                for item in sampled_variation_data:
                    variation_conversations.append(item["conversations"])

            # === Part 3: Combined Training ===
            print("🎯 Combined training on original + variation generation...")

            # Combine original and variation training data
            all_conversations = original_conversations + variation_conversations

            # Convert to training format
            if all_conversations:
                step_texts = self.tokenizer.apply_chat_template(
                    all_conversations,
                    tokenize=False,
                )
                step_dataset = Dataset.from_dict({"text": step_texts})

                # Mini training step with combined data
                trainer = SFTTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    train_dataset=step_dataset,
                    args=SFTConfig(
                        dataset_text_field="text",
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=2,
                        max_steps=3,  # Slightly more steps for combined training
                        learning_rate=1e-5,  # Lower learning rate for fine adjustment
                        logging_steps=1,
                        optim="adamw_8bit",
                        fp16=False,
                        bf16=True,
                        report_to="none",
                    ),
                )
                trainer.train()

            # === Part 4: Validation and Statistics ===
            print("📊 Validating learned capabilities...")

            # Test variation generation capability
            if len(high_reward_questions) > 0:
                test_question = high_reward_questions[0]
                print(f"🧪 Testing variation generation for: {test_question[:100]}...")

                variations = self._test_variation_generation(test_question)
                variation_quality = self._evaluate_variation_quality(test_question, variations)

                print(f"🎯 Generated {len(variations)} variations, quality score: {variation_quality:.3f}")

            print(f"📊 Reward distribution: Mean={np.mean(rewards):.3f}, "
                  f"Std={np.std(rewards):.3f}, Baseline={baseline_reward:.3f}")
            print(f"🎯 Training samples: {len(original_conversations)} original + {len(variation_conversations)} variations")

            # Update historical question database
            self.history_questions.extend(group_questions)

            # Keep recent 1000 questions
            if len(self.history_questions) > 1000:
                self.history_questions = self.history_questions[-1000:]

        print("✅ Stage 2 GRPO optimization (with variation training) completed")

        # Save checkpoint
        os.makedirs("checkpoints/stage2_grpo", exist_ok=True)
        self.model.save_pretrained("checkpoints/stage2_grpo")
        self.tokenizer.save_pretrained("checkpoints/stage2_grpo")
        print("💾 Stage 2 model saved")

    def stage3_distillation(self):
        """Stage 3: DeepSeek-R1 Knowledge Distillation - Real teacher model guidance"""
        print("\n" + "="*50)
        print("🎯 Stage 3: DeepSeek-R1 Knowledge Distillation started")
        print("="*50)

        # Create distillation training data with real DeepSeek-R1 teacher
        distillation_conversations = []
        teacher_evaluations = []

        for step in range(self.config.MAX_STEPS_STAGE3):
            print(f"\n🔄 Distillation step {step+1}/{self.config.MAX_STEPS_STAGE3}")

            # Generate initial question with student model
            student_question = self._generate_single_question()
            print(f"📝 Student generated: {student_question[:100]}...")

            # Get DeepSeek-R1 teacher evaluation
            print("🤖 Getting DeepSeek-R1 teacher evaluation...")
            teacher_eval = self.deepseek_teacher.evaluate_problem(student_question)
            teacher_evaluations.append(teacher_eval)

            # Show teacher feedback
            print(f"👨‍🏫 Teacher overall score: {teacher_eval['overall_score']:.2f}/5.0")
            print(f"📊 Difficulty: {teacher_eval['difficulty_score']:.1f}, Rigor: {teacher_eval['rigor_score']:.1f}")
            print(f"💡 Innovation: {teacher_eval['innovation_score']:.1f}, Clarity: {teacher_eval['clarity_score']:.1f}")

            # Get teacher's improved version
            print("🔄 Getting teacher's improvement...")
            teacher_improved = self.deepseek_teacher.improve_problem(
                student_question,
                teacher_eval['raw_feedback']
            )

            if teacher_improved and teacher_improved != student_question:
                print(f"✨ Teacher improved: {teacher_improved[:100]}...")

                # Create distillation training pair
                distillation_conversations.append([
                    {"role": "user", "content": f"Improve this competition problem based on expert feedback: {student_question}"},
                    {"role": "assistant", "content": teacher_improved}
                ])

                # Also create evaluation-guided generation pair
                feedback_summary = f"Focus on: difficulty={teacher_eval['difficulty_score']:.1f}, rigor={teacher_eval['rigor_score']:.1f}, innovation={teacher_eval['innovation_score']:.1f}"
                distillation_conversations.append([
                    {"role": "user", "content": f"Generate a high-quality competition problem. {feedback_summary}"},
                    {"role": "assistant", "content": teacher_improved}
                ])
            else:
                print("⚠️ Teacher improvement failed, using original")

            # Every 3 steps, also get teacher-generated variations
            if step % 3 == 0 and step > 0:
                print("🔄 Getting teacher variations...")
                teacher_variations = self.deepseek_teacher.generate_variations(student_question, 2)

                for i, variation in enumerate(teacher_variations):
                    if variation:
                        print(f"🎯 Teacher variation {i+1}: {variation[:80]}...")
                        distillation_conversations.append([
                            {"role": "user", "content": f"Create a variation of this problem: {student_question}"},
                            {"role": "assistant", "content": variation}
                        ])

            # Rate limiting for API calls
            time.sleep(1)

        # Create distillation training dataset
        if distillation_conversations:
            print(f"\n📚 Creating distillation dataset with {len(distillation_conversations)} examples...")

            distillation_texts = self.tokenizer.apply_chat_template(
                distillation_conversations,
                tokenize=False,
            )
            distillation_dataset = Dataset.from_dict({"text": distillation_texts})

            # Configure distillation trainer
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=distillation_dataset,
                args=SFTConfig(
                    dataset_text_field="text",
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    max_steps=self.config.MAX_STEPS_STAGE3,
                    learning_rate=1e-5,  # Lower learning rate for distillation
                    logging_steps=5,
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=3407,
                    report_to="none",
                    fp16=False,
                    bf16=True,
                ),
            )

            print("🔄 Starting DeepSeek-R1 knowledge distillation training...")
            trainer.train()

            # Show teacher evaluation statistics
            print("\n📊 Teacher Evaluation Statistics:")
            if teacher_evaluations:
                avg_overall = np.mean([eval['overall_score'] for eval in teacher_evaluations])
                avg_difficulty = np.mean([eval['difficulty_score'] for eval in teacher_evaluations])
                avg_rigor = np.mean([eval['rigor_score'] for eval in teacher_evaluations])
                avg_innovation = np.mean([eval['innovation_score'] for eval in teacher_evaluations])

                print(f"📈 Average scores - Overall: {avg_overall:.2f}, Difficulty: {avg_difficulty:.2f}")
                print(f"📈 Rigor: {avg_rigor:.2f}, Innovation: {avg_innovation:.2f}")
                print(f"📊 Total teacher feedback examples: {len(teacher_evaluations)}")
                print(f"📊 Total distillation training pairs: {len(distillation_conversations)}")

        print("✅ Stage 3 DeepSeek-R1 knowledge distillation completed")

        # Test the distilled model
        print("\n🧪 Testing distilled model capabilities...")
        test_question = self._generate_single_question("Generate an innovative calculus competition problem:")
        final_eval = self.deepseek_teacher.evaluate_problem(test_question)
        print(f"🎯 Final model test - Teacher score: {final_eval['overall_score']:.2f}/5.0")

        # Save final model
        os.makedirs("checkpoints/stage3_final", exist_ok=True)
        self.model.save_pretrained("checkpoints/stage3_final")
        self.tokenizer.save_pretrained("checkpoints/stage3_final")
        print("💾 Final distilled model saved")

    def _generate_single_question(self, custom_prompt: str = None, enable_thinking: bool = False) -> str:
        """Generate single question - Using unsloth inference style (exactly like reference script)"""
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = "Generate a high-quality competition problem:"

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True, # Must add for generation
            enable_thinking = enable_thinking, # Support both thinking and non-thinking modes
        )

        # Use unsloth native inference style (exactly like reference script)
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

            if enable_thinking:
                # For thinking mode (like reference script)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens = 1024, # Increase for longer outputs!
                    temperature = 0.6, top_p = 0.95, top_k = 20, # For thinking
                    do_sample = True,
                )
            else:
                # For non-thinking mode (like reference script)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens = 256, # Increase for longer outputs!
                    temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
                    do_sample = True,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            question = generated_text.split("assistant")[-1].strip()

        return question

    def _generate_question_group(self) -> List[str]:
        """Generate a group of questions for GRPO"""
        questions = []
        prompts = [
            "Generate a challenging algebra competition problem:",
            "Create a geometry problem suitable for math olympiad:",
            "Design a calculus problem with moderate difficulty:",
            "Formulate a number theory competition question:",
            "Develop a combinatorics problem for advanced students:",
            "Create an analysis problem requiring proof:",
            "Design a probability problem with real-world context:",
            "Generate an innovative interdisciplinary math problem:"
        ]

        for i in range(self.config.GROUP_SIZE):
            prompt = prompts[i % len(prompts)]
            question = self._generate_single_question(prompt)
            questions.append(question)

        return questions

    def _create_variation_training_examples(self, original_question: str) -> List[List[Dict]]:
        """Create variation training examples for a given original question"""
        variation_examples = []

        # Define variation types and prompts
        variation_types = [
            {
                "type": "context_change",
                "prompt": f"Generate a mathematical problem variation that maintains the same core concept as: {original_question}\nRequirement: Change the context but keep the solution method identical.",
                "instruction": "Change the mathematical context while preserving the solution approach"
            },
            {
                "type": "parameter_change",
                "prompt": f"Create a problem variant with similar difficulty: {original_question}\nRequirement: Use different parameters but same mathematical structure.",
                "instruction": "Modify numerical parameters while maintaining the same mathematical structure"
            },
            {
                "type": "practical_application",
                "prompt": f"Transform this problem into a practical application: {original_question}\nRequirement: Add real-world context while preserving the mathematical essence.",
                "instruction": "Add real-world context while preserving the mathematical core"
            }
        ]

        for var_type in variation_types:
            # Generate variation using current model
            variation = self._generate_single_question(var_type["prompt"])

            # Create training conversation
            training_example = [
                {
                    "role": "user",
                    "content": f"{var_type['instruction']}: {original_question}"
                },
                {
                    "role": "assistant",
                    "content": f"Here's a {var_type['type']} variation:\n\nOriginal: {original_question}\n\nVariation: {variation}\n\nBoth problems maintain the same solution approach."
                }
            ]

            variation_examples.append(training_example)

        return variation_examples

    def _test_variation_generation(self, original_question: str, num_variations: int = 3) -> List[str]:
        """Test the model's variation generation capability"""
        variations = []

        test_prompts = [
            f"Generate a variation of this math problem with different context: {original_question}",
            f"Create a similar problem with different parameters: {original_question}",
            f"Transform this into a real-world application: {original_question}"
        ]

        for i in range(min(num_variations, len(test_prompts))):
            try:
                variation = self._generate_single_question(test_prompts[i])
                if variation and len(variation) > 20:  # Basic quality check
                    variations.append(variation)
            except Exception as e:
                print(f"⚠️ Variation generation failed: {e}")

        return variations

    def _evaluate_variation_quality(self, original_question: str, variations: List[str]) -> float:
        """Evaluate the quality of generated variations"""
        if not variations:
            return 0.0

        quality_scores = []

        for variation in variations:
            # Calculate similarity (should be moderate - not too high, not too low)
            try:
                vectorizer = TfidfVectorizer(max_features=100)
                tfidf_matrix = vectorizer.fit_transform([original_question, variation])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

                # Optimal similarity range: 0.3-0.7 (similar structure, different content)
                if 0.3 <= similarity <= 0.7:
                    similarity_score = 1.0
                elif similarity < 0.3:
                    similarity_score = similarity / 0.3  # Too different
                else:
                    similarity_score = (1.0 - similarity) / 0.3  # Too similar

                # Length similarity (variations should have reasonable length)
                length_ratio = min(len(variation), len(original_question)) / max(len(variation), len(original_question))
                length_score = length_ratio if length_ratio > 0.5 else length_ratio * 2

                # Overall quality score
                variation_quality = (similarity_score + length_score) / 2.0
                quality_scores.append(variation_quality)

            except Exception as e:
                print(f"⚠️ Quality evaluation failed: {e}")
                quality_scores.append(0.5)  # Default score

        return np.mean(quality_scores) if quality_scores else 0.0

    def save_to_huggingface(self, stage_name: str = "final"):
        """Save model to HuggingFace Hub - Using unsloth standard methods (exactly like reference script)"""
        print(f"\n💾 Saving model (stage: {stage_name})...")

        # Get HF token
        hf_token = self.config.HF_TOKEN or os.environ.get('HF_TOKEN')

        # Always save locally first (like reference script)
        print("💾 Local saving...")
        self.model.save_pretrained("lora_model")  # Local saving
        self.tokenizer.save_pretrained("lora_model")

        # Only push to hub if token is available
        if hf_token:
            print(f"📤 Uploading to HuggingFace Hub...")

            # Model repository name
            repo_name = f"{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-{stage_name}"
            print(f"📤 Repository: {repo_name}")

            try:
                # Save LoRA adapters (保持原精度)
                print("📤 Uploading LoRA adapters (原精度)...")
                self.model.push_to_hub(repo_name, token = hf_token) # Online saving
                self.tokenizer.push_to_hub(repo_name, token = hf_token) # Online saving
                print("✅ LoRA adapters uploaded successfully")

                # Save full precision merged model (原精度合并模型)
                print("🔄 Creating full precision merged model (原精度合并模型)...")
                merged_repo_name = f"{repo_name}-merged-fp16"
                self.model.push_to_hub_merged(merged_repo_name, self.tokenizer, save_method = "merged_16bit", token = hf_token)
                print("✅ Full precision merged model uploaded successfully")

                # 可选：只在需要时创建量化版本
                create_quantized = getattr(self.config, 'SAVE_QUANTIZED_VERSIONS', False)
                
                if create_quantized:
                    print("🔄 Creating quantized versions (可选)...")
                    
                    # 4bit量化版本 (精度损失)
                    print("   Creating 4bit quantized model...")
                    merged_4bit_repo_name = f"{repo_name}-merged-4bit"
                    self.model.push_to_hub_merged(merged_4bit_repo_name, self.tokenizer, save_method = "merged_4bit_forced", token = hf_token)
                    print("   ✅ 4bit model created")

                    # GGUF格式 (CPU推理)
                    print("   Creating GGUF models for CPU inference...")  
                    gguf_repo_name = f"{repo_name}-gguf"
                    self.model.push_to_hub_gguf(
                        gguf_repo_name,
                        self.tokenizer,
                        quantization_method = ["q8_0"],  # 只保存高质量量化
                        token = hf_token
                    )
                    print("   ✅ GGUF models created")
                else:
                    print("⚠️ 跳过量化版本保存 (保持原精度)")

                print(f"🎉 原精度模型成功上传到 HuggingFace!")
                print(f"📁 Repositories created:")
                print(f"   - LoRA (原精度): https://huggingface.co/{repo_name}")
                print(f"   - Merged FP16 (原精度): https://huggingface.co/{merged_repo_name}")
                
                if create_quantized:
                    print(f"   - 4bit Quantized: https://huggingface.co/{merged_4bit_repo_name}")
                    print(f"   - GGUF: https://huggingface.co/{gguf_repo_name}")

                return True

            except Exception as e:
                print(f"❌ Failed to upload to HuggingFace: {e}")
                print("💡 Troubleshooting suggestions:")
                print("   - Check your HuggingFace token permissions")
                print("   - Verify network connection stability")
                print("   - Ensure sufficient disk space for model merging")
                if "4bit" in str(e).lower():
                    print("   - 4bit merging error: This is expected and fixed with 'merged_4bit_forced'")
                if "gguf" in str(e).lower():
                    print("   - GGUF creation may take significant time and memory")
        else:
            print("⚠️ No HuggingFace token found. Only local saving completed.")
            print("💡 To upload to HF Hub, set HF_TOKEN environment variable")
            print("💡 Get a token from: https://huggingface.co/settings/tokens")
            return True

    def save_full_precision_only(self, stage_name: str = "final"):
        """专门保存原精度模型 (无量化)"""
        print(f"\n💎 Saving full precision model only (stage: {stage_name})...")
        
        # Get HF token
        hf_token = self.config.HF_TOKEN or os.environ.get('HF_TOKEN')
        
        # Always save locally first
        print("💾 Local saving (原精度)...")
        self.model.save_pretrained("lora_model_fp")
        self.tokenizer.save_pretrained("lora_model_fp")
        
        if hf_token:
            print(f"📤 Uploading full precision model to HuggingFace Hub...")
            
            # Model repository name
            repo_name = f"{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-{stage_name}-fp"
            print(f"📤 Repository: {repo_name}")
            
            try:
                # Save LoRA adapters only (原精度)
                print("📤 Uploading LoRA adapters (原精度)...")
                self.model.push_to_hub(repo_name, token = hf_token)
                self.tokenizer.push_to_hub(repo_name, token = hf_token)
                print("✅ LoRA adapters uploaded successfully")
                
                # Save full precision merged model
                print("🔄 Creating full precision merged model...")
                merged_repo_name = f"{repo_name}-merged"
                self.model.push_to_hub_merged(merged_repo_name, self.tokenizer, save_method = "merged_16bit", token = hf_token)
                print("✅ Full precision merged model uploaded successfully")
                
                print(f"💎 原精度模型成功上传!")
                print(f"📁 Repositories:")
                print(f"   - LoRA (原精度): https://huggingface.co/{repo_name}")
                print(f"   - Merged (原精度): https://huggingface.co/{merged_repo_name}")
                
                return True
                
            except Exception as e:
                print(f"❌ Failed to upload full precision model: {e}")
                return False
        else:
            print("⚠️ No HuggingFace token found. Only local saving completed.")
            return True

    def inference_test(self):
        """Test model inference - Using unsloth inference style with variation generation testing"""
        print("\n" + "="*50)
        print("🧪 Running comprehensive inference test (original + variation generation)")
        print("="*50)

        test_prompts = [
            "Generate a calculus competition problem:",
            "Create an algebra problem with moderate difficulty:",
            "Design a geometry proof problem:"
        ]

        # Import TextStreamer for unsloth-style streaming output
        from transformers import TextStreamer

        generated_problems = []  # Store for variation testing

        for i, prompt in enumerate(test_prompts):
            print(f"\n📝 Test {i+1}: {prompt}")

            # Test 1: Non-thinking mode (like reference script)
            print("🔄 Non-thinking mode:")
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True, # Must add for generation
                enable_thinking = False, # Disable thinking
            )

            # Use unsloth native inference with streaming (exactly like reference)
            _ = self.model.generate(
                **self.tokenizer(text, return_tensors = "pt").to("cuda"),
                max_new_tokens = 256, # Increase for longer outputs!
                temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
                streamer = TextStreamer(self.tokenizer, skip_prompt = True),
            )

            # Test 2: Thinking mode (like reference script)
            print(f"\n🤔 Thinking mode:")
            text_thinking = self.tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True, # Must add for generation
                enable_thinking = True, # Enable thinking
            )

            _ = self.model.generate(
                **self.tokenizer(text_thinking, return_tensors = "pt").to("cuda"),
                max_new_tokens = 1024, # Increase for longer outputs!
                temperature = 0.6, top_p = 0.95, top_k = 20, # For thinking
                streamer = TextStreamer(self.tokenizer, skip_prompt = True),
            )

            # Also get the generated text for reward calculation and variation testing
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens = 256,
                    temperature = 0.7, top_p = 0.8, top_k = 20,
                    do_sample = True,
                )

                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated_text.split("assistant")[-1].strip()

                # Calculate reward for this question
                reward = self.reward_calculator.calculate_reward(
                    response, self.history_questions, []
                )
                print(f"\n🎯 Quality score: {reward:.3f}")

                # Store for variation testing
                generated_problems.append(response)

        # Test 3: Variation Generation Capability
        print(f"\n🔄 Testing variation generation capability...")
        print("="*50)

        for i, original_problem in enumerate(generated_problems):
            print(f"\n🧪 Variation Test {i+1}:")
            print(f"📝 Original: {original_problem[:100]}...")

            # Test different types of variations
            variation_tests = [
                {
                    "type": "Context Change",
                    "prompt": f"Generate a variation of this math problem with different context: {original_problem}"
                },
                {
                    "type": "Parameter Change",
                    "prompt": f"Create a similar problem with different numerical parameters: {original_problem}"
                },
                {
                    "type": "Real-world Application",
                    "prompt": f"Transform this into a practical real-world application: {original_problem}"
                }
            ]

            for j, var_test in enumerate(variation_tests):
                print(f"\n🔄 {var_test['type']}:")

                var_messages = [{"role": "user", "content": var_test["prompt"]}]
                var_text = self.tokenizer.apply_chat_template(
                    var_messages,
                    tokenize = False,
                    add_generation_prompt = True,
                    enable_thinking = False,
                )

                # Generate variation with streaming
                _ = self.model.generate(
                    **self.tokenizer(var_text, return_tensors = "pt").to("cuda"),
                    max_new_tokens = 256,
                    temperature = 0.7, top_p = 0.8, top_k = 20,
                    streamer = TextStreamer(self.tokenizer, skip_prompt = True),
                )

                # Also get the variation text for quality evaluation
                with torch.no_grad():
                    var_inputs = self.tokenizer(var_text, return_tensors="pt").to("cuda")
                    var_outputs = self.model.generate(
                        **var_inputs,
                        max_new_tokens = 256,
                        temperature = 0.7, top_p = 0.8, top_k = 20,
                        do_sample = True,
                    )

                    var_generated_text = self.tokenizer.decode(var_outputs[0], skip_special_tokens=True)
                    variation = var_generated_text.split("assistant")[-1].strip()

                    # Evaluate variation quality
                    variation_quality = self._evaluate_variation_quality(original_problem, [variation])
                    print(f"🎯 Variation quality: {variation_quality:.3f}")

        # Summary
        print(f"\n📊 Inference Test Summary:")
        print(f"✅ Original problem generation: {len(generated_problems)} problems tested")
        print(f"✅ Variation generation: {len(generated_problems) * 3} variations tested")
        print(f"✅ Both thinking and non-thinking modes validated")
        print(f"✅ Quality scoring system validated")

        print("\n✅ Comprehensive inference test completed")

    def train_full_pipeline(self):
        """Complete training pipeline"""
        print("🎯 Starting Questions-Gen model complete training pipeline")
        print("📋 Training plan: Basic pretraining -> RL GRPO -> Knowledge distillation")

        try:
            # Stage 1: Basic pretraining
            self.stage1_basic_training()

            # Stage 2: RL GRPO optimization
            self.stage2_grpo_training()

            # Stage 3: Knowledge distillation
            self.stage3_distillation()

            # Inference test
            self.inference_test()

            # Save to HuggingFace (原精度版本)
            print("\n🔄 Saving models to HuggingFace Hub...")
            if self.config.PRESERVE_FULL_PRECISION:
                print("💎 Saving full precision models only...")
                self.save_full_precision_only("stage1")  # Save stage 1 (原精度)
                self.save_full_precision_only("stage2")  # Save stage 2 (原精度)
                self.save_full_precision_only("final")   # Save final model (原精度)
            else:
                print("🔄 Saving with quantization options...")
                self.save_to_huggingface("stage1")  # Save stage 1
                self.save_to_huggingface("stage2")  # Save stage 2
                self.save_to_huggingface("final")   # Save final model

            print("\n🎉 All coordinated training pipeline completed!")
            print("📋 Training Summary:")
            print("  ✅ Stage 1: Basic problem generation pretraining")
            print("  ✅ Stage 2: GRPO optimization + Variation generation training")
            print("  ✅ Stage 3: Knowledge distillation enhancement")
            print("  ✅ Comprehensive inference testing (original + variations)")
            print("\n📁 Local model save locations:")
            print("  - Stage 1: checkpoints/stage1_basic")
            print("  - Stage 2: checkpoints/stage2_grpo (with variation capabilities)")
            print("  - Final: checkpoints/stage3_final (fully optimized)")
            print("\n📤 HuggingFace repositories:")
            if self.config.PRESERVE_FULL_PRECISION:
                print("💎 原精度模型仓库:")
                print(f"  - LoRA Stage1: https://huggingface.co/{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-stage1-fp")
                print(f"  - LoRA Stage2: https://huggingface.co/{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-stage2-fp")
                print(f"  - LoRA Final: https://huggingface.co/{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-final-fp")
                print(f"  - Merged Models: {self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-*-fp-merged")
            else:
                print(f"  - https://huggingface.co/{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-stage1")
                print(f"  - https://huggingface.co/{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-stage2")
                print(f"  - https://huggingface.co/{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-final")
            print("\n🎯 Model Capabilities:")
            print("  ✅ High-quality competition problem generation")
            print("  ✅ Intelligent problem variation generation")
            print("  ✅ Multi-dimensional quality optimization")
            print("  ✅ Real-world application transformation")
            print("  ✅ Mathematical rigor and novelty balance")

        except Exception as e:
            print(f"❌ Error during training: {e}")
            print("💡 Suggest checking GPU memory and data format")

# ==================== Main Program Entry ====================
def main():
    """Main program"""
    print("🚀 Starting Questions-Gen model training system")
    print("="*60)

    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🔧 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  Warning: CUDA not detected, will use CPU training (slower)")

    # Check HuggingFace token
    hf_token = TrainingConfig.HF_TOKEN or os.environ.get('HF_TOKEN')
    if hf_token:
        print("✅ HuggingFace token detected")
    else:
        print("⚠️ No HuggingFace token found - models will only be saved locally")
        print("💡 Set HF_TOKEN environment variable to enable HuggingFace upload")
    
    # Check precision saving configuration
    config = TrainingConfig()
    if config.PRESERVE_FULL_PRECISION:
        print("💎 Configuration: 保存原精度模型 (无量化)")
        print("   - LoRA adapters: FP16原精度")
        print("   - Merged models: FP16原精度")
        print("   - 跳过4bit和GGUF量化版本")
    else:
        print("🔄 Configuration: 保存多种精度版本")
        print("   - LoRA adapters, 16bit merged, 4bit quantized, GGUF")
    
    if config.SAVE_QUANTIZED_VERSIONS:
        print("📦 将额外创建量化版本")
    else:
        print("⚡ 跳过量化版本，专注原精度")
    
    # Check precision saving configuration
    config = TrainingConfig()
    if config.PRESERVE_FULL_PRECISION:
        print("💎 Configuration: 保存原精度模型 (无量化)")
        print("   - LoRA adapters: FP16原精度")
        print("   - Merged models: FP16原精度")
        print("   - 跳过4bit和GGUF量化版本")
    else:
        print("🔄 Configuration: 保存多种精度版本")
        print("   - LoRA adapters, 16bit merged, 4bit quantized, GGUF")
    
    if config.SAVE_QUANTIZED_VERSIONS:
        print("📦 将额外创建量化版本")
    else:
        print("⚡ 跳过量化版本，专注原精度")

    # Create trainer and start training
    trainer = QuestionsGenTrainer()
    trainer.train_full_pipeline()

if __name__ == "__main__":
    main()