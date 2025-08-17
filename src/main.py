from src.QuestionsGenTrainer import QuestionsGenTrainer
from src.config import TrainingConfig
import torch
import os

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

    # Create trainer and start training
    trainer = QuestionsGenTrainer()
    trainer.train_full_pipeline()
