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
