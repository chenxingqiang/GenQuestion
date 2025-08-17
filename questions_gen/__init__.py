# -*- coding: utf-8 -*-
"""
Questions-Gen: AI-Powered Competition Problem Generation Package

An advanced package for generating high-quality mathematical competition problems
using fine-tuned language models with reinforcement learning optimization.
"""

__version__ = "0.1.1"
__author__ = "xingqiang"
__email__ = "your.email@turingai.cc"

# Core components - lazy loading to avoid import issues
def _lazy_import():
    """Lazy import to avoid circular dependencies."""
    try:
        from .core.config import TrainingConfig
        from .models.reward_calculator import RewardCalculator
        from .models.novelty_constraint import NoveltyConstraint
        from .models.deepseek_teacher import DeepSeekTeacher
        return {
            "TrainingConfig": TrainingConfig,
            "RewardCalculator": RewardCalculator, 
            "NoveltyConstraint": NoveltyConstraint,
            "DeepSeekTeacher": DeepSeekTeacher,
        }
    except ImportError as e:
        print(f"⚠️ Some dependencies not available: {e}")
        return {}

# Heavy imports only when needed
def get_trainer():
    """Get QuestionsGenTrainer with full dependencies."""
    from .core.trainer import QuestionsGenTrainer
    return QuestionsGenTrainer

def get_data_preparer():
    """Get QuestionsDataPreparer with full dependencies."""
    from .data.data_preparer import QuestionsDataPreparer
    return QuestionsDataPreparer

def train_full_pipeline(config=None):
    """Train the full pipeline with dependencies check."""
    from .core.trainer import train_full_pipeline as _train
    return _train(config)

# Load what we can
_available = _lazy_import()

# Export available components
__all__ = list(_available.keys()) + [
    "get_trainer", "get_data_preparer", "train_full_pipeline"
]

# Make available components accessible
globals().update(_available)
