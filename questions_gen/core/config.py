# -*- coding: utf-8 -*-
"""Training configuration parameters."""

class TrainingConfig:
    """Questions-Gen model training configuration."""
    
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
