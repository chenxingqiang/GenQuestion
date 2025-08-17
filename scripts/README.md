# Training Scripts

This directory contains training and development scripts for Questions-Gen.

## 🎯 Training Scripts

### Main Training Script

- **`questions_gen_training.py`** - Complete three-stage training pipeline
  ```bash
  python scripts/questions_gen_training.py
  ```

## 📋 Training Pipeline

The main training script implements a comprehensive three-stage approach:

### Stage 1: Basic Pretraining
- Supervised fine-tuning on mathematical datasets
- LoRA efficient parameter updates
- Foundation model capabilities

### Stage 2: RL GRPO Optimization
- Group Policy Reinforcement Learning
- Multi-dimensional reward optimization
- Variation generation training

### Stage 3: Knowledge Distillation
- DeepSeek-R1 teacher model integration
- Expert guidance and improvement
- Quality enhancement through distillation

## 🔧 Requirements

### Hardware Requirements
- **GPU**: 8GB+ VRAM (recommended for Qwen3-14B 4bit)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ available space

### Software Dependencies
```bash
# Core ML frameworks
pip install torch>=2.0.0
pip install transformers>=4.35.0
pip install unsloth[colab-new]>=2024.1

# Training frameworks
pip install datasets>=2.14.0
pip install trl>=0.7.0
pip install accelerate>=0.24.0

# For DeepSeek teacher model
pip install openai>=1.0.0
```

## ⚙️ Configuration

### Environment Variables
```bash
# HuggingFace token for model uploads
export HF_TOKEN="your_huggingface_token"

# DeepSeek API key for teacher model
export DEEPSEEK_API_KEY="your_deepseek_api_key"

# Optional: HuggingFace mirror for China
export HF_ENDPOINT="https://hf-mirror.com"
```

### Training Configuration
Edit the `TrainingConfig` class in the script to customize:

```python
class TrainingConfig:
    # Model settings
    MODEL_NAME = "unsloth/Qwen3-14B"
    MAX_SEQ_LENGTH = 2048
    
    # Training steps
    MAX_STEPS_STAGE1 = 200  # Basic pretraining
    MAX_STEPS_STAGE2 = 100  # RL GRPO
    MAX_STEPS_STAGE3 = 80   # Knowledge distillation
    
    # GRPO settings
    GROUP_SIZE = 8
    REWARD_WEIGHTS = {
        'difficulty': 0.4,
        'novelty': 0.3,
        'rigor': 0.2,
        'diversity': 0.1
    }
```

## 🚀 Usage Examples

### Basic Training
```bash
# Run complete training pipeline
python scripts/questions_gen_training.py
```

### Custom Training (modify the script)
```python
# In questions_gen_training.py
def main():
    trainer = QuestionsGenTrainer()
    
    # Run individual stages
    trainer.stage1_basic_training()
    trainer.stage2_grpo_training()
    trainer.stage3_distillation()
    
    # Test the model
    trainer.inference_test()
```

## 📊 Output

Training creates the following outputs:

### Model Checkpoints
```
checkpoints/
├── stage1_basic/     # Stage 1 model
├── stage2_grpo/      # Stage 2 GRPO model
└── stage3_final/     # Final distilled model
```

### HuggingFace Repositories
Automatically uploaded if `HF_TOKEN` is set:
- `xingqiang/questions-gen-qwen3-14b-stage1`
- `xingqiang/questions-gen-qwen3-14b-stage2`
- `xingqiang/questions-gen-qwen3-14b-final`

## 🔍 Monitoring

The script provides comprehensive monitoring:
- GPU memory usage tracking
- Training progress validation
- Quality score assessment
- Teacher model feedback analysis

## ⚠️ Troubleshooting

### Common Issues

1. **CUDA Memory Error**
   - Reduce `BATCH_SIZE` or `GROUP_SIZE`
   - Enable more aggressive memory optimization

2. **Network Connection Issues**
   - Set `HF_ENDPOINT` for mirror access
   - Check firewall settings for HuggingFace/DeepSeek

3. **API Rate Limiting**
   - Adjust sleep intervals in DeepSeek calls
   - Use alternative teacher models if needed

### Memory Optimization
The script includes automatic memory management:
- GPU memory monitoring
- Automatic cache clearing
- Conservative batch sizing fallbacks

## 📚 Learn More

- [Training Guide](../docs/guides/TRAINING_GUIDE.md)
- [Technical Details](../docs/technical/TRAINING_DETAILS.md)
- [Optimization Guide](../docs/training/OPTIMIZATION_SUMMARY.md)
