# -*- coding: utf-8 -*-
"""Main trainer class for Questions-Gen model."""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import TextStreamer
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import TrainingConfig
from ..models.reward_calculator import RewardCalculator
from ..models.novelty_constraint import NoveltyConstraint
from ..models.deepseek_teacher import DeepSeekTeacher
from ..data.data_preparer import QuestionsDataPreparer


class QuestionsGenTrainer:
    """Questions-Gen model trainer with three-stage training pipeline."""

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
        """监控GPU内存使用."""
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
        """统一的注意力偏置修复方法."""
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
        """验证训练进度."""
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
        """Load and configure model - Using unsloth standard approach."""
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

    def _generate_single_question(self, custom_prompt: str = None, enable_thinking: bool = False) -> str:
        """Generate single question - Using unsloth inference style."""
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

        # Use unsloth native inference style
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

            if enable_thinking:
                # For thinking mode
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens = 1024, # Increase for longer outputs!
                    temperature = 0.6, top_p = 0.95, top_k = 20, # For thinking
                    do_sample = True,
                )
            else:
                # For non-thinking mode
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
        """Generate a group of questions for GRPO."""
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
        """Create variation training examples for a given original question."""
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
        """Test the model's variation generation capability."""
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
        """Evaluate the quality of generated variations."""
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


# Convenience function for easy import
def train_full_pipeline(config: TrainingConfig = None):
    """Convenience function to train the full pipeline."""
    if config:
        # Update global config if provided
        pass
    
    trainer = QuestionsGenTrainer()
    trainer.train_full_pipeline()
    return trainer
