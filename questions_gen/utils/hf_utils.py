# -*- coding: utf-8 -*-
"""HuggingFace utilities for Questions-Gen models."""

import os
import json
import requests
import pandas as pd
from typing import Dict, List, Optional
from huggingface_hub import HfApi, list_models, model_info


class HuggingFaceUtils:
    """HuggingFace相关工具。"""

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get('HF_TOKEN')
        self.api = HfApi(token=self.token) if self.token else HfApi()
        
        # Questions-Gen模型信息
        self.qgen_models = {
            "stage1": "xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged",
            "stage2": "xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit",
            "final": "xingqiang/questions-gen-qwen3-14b-final-merged-16bit"
        }

    def verify_models_exist(self) -> Dict[str, bool]:
        """验证所有模型是否存在于HuggingFace上。"""
        print("🔍 Verifying Questions-Gen models on HuggingFace...")
        
        verification_results = {}
        
        for stage, model_name in self.qgen_models.items():
            try:
                info = model_info(model_name)
                verification_results[stage] = True
                print(f"✅ {stage}: {model_name}")
                print(f"   📊 Downloads: {info.downloads if hasattr(info, 'downloads') else 'N/A'}")
                print(f"   📅 Last modified: {info.lastModified if hasattr(info, 'lastModified') else 'N/A'}")
            except Exception as e:
                verification_results[stage] = False
                print(f"❌ {stage}: {model_name} - Error: {e}")

        return verification_results

    def get_model_details(self, model_stage: str) -> Optional[Dict]:
        """获取模型详细信息。"""
        if model_stage not in self.qgen_models:
            print(f"❌ Unknown model stage: {model_stage}")
            return None

        model_name = self.qgen_models[model_stage]
        
        try:
            info = model_info(model_name)
            
            model_details = {
                "model_name": model_name,
                "stage": model_stage,
                "id": info.id,
                "downloads": getattr(info, 'downloads', 0),
                "likes": getattr(info, 'likes', 0),
                "last_modified": str(getattr(info, 'lastModified', 'Unknown')),
                "pipeline_tag": getattr(info, 'pipeline_tag', 'text-generation'),
                "tags": getattr(info, 'tags', []),
                "siblings": [f.rfilename for f in info.siblings] if hasattr(info, 'siblings') else [],
                "model_size_gb": self._estimate_model_size(info),
                "library_name": getattr(info, 'library_name', 'transformers')
            }
            
            return model_details
            
        except Exception as e:
            print(f"❌ Error getting model details for {model_name}: {e}")
            return None

    def _estimate_model_size(self, model_info) -> Optional[float]:
        """估算模型大小（GB）。"""
        try:
            if hasattr(model_info, 'siblings'):
                total_size = 0
                for file_info in model_info.siblings:
                    if hasattr(file_info, 'size') and file_info.size:
                        total_size += file_info.size
                
                if total_size > 0:
                    return round(total_size / (1024**3), 2)  # Convert to GB
            
            return None
        except:
            return None

    def compare_all_models(self) -> Dict:
        """比较所有Questions-Gen模型。"""
        print("📊 Comparing all Questions-Gen models on HuggingFace...")
        print("="*60)
        
        comparison = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "models": {},
            "comparison_metrics": {}
        }
        
        # Get details for each model
        for stage in self.qgen_models.keys():
            details = self.get_model_details(stage)
            if details:
                comparison["models"][stage] = details

        # Generate comparison metrics
        valid_models = comparison["models"]
        if len(valid_models) > 1:
            # Downloads comparison
            downloads = {stage: details["downloads"] for stage, details in valid_models.items()}
            most_downloaded = max(downloads, key=downloads.get) if downloads else None
            
            # Likes comparison  
            likes = {stage: details["likes"] for stage, details in valid_models.items()}
            most_liked = max(likes, key=likes.get) if likes else None
            
            # Size comparison
            sizes = {stage: details["model_size_gb"] for stage, details in valid_models.items() 
                    if details["model_size_gb"] is not None}
            
            comparison["comparison_metrics"] = {
                "most_downloaded": most_downloaded,
                "most_liked": most_liked,
                "download_stats": downloads,
                "like_stats": likes,
                "size_stats": sizes
            }

        self._print_model_comparison(comparison)
        return comparison

    def _print_model_comparison(self, comparison: Dict):
        """打印模型比较结果。"""
        print("\n📋 Model Comparison Results:")
        print("-" * 40)
        
        for stage, details in comparison["models"].items():
            print(f"\n🎯 {stage.upper()} Model:")
            print(f"   📦 Name: {details['model_name']}")
            print(f"   📊 Downloads: {details['downloads']:,}")
            print(f"   👍 Likes: {details['likes']}")
            if details['model_size_gb']:
                print(f"   💾 Size: {details['model_size_gb']} GB")
            print(f"   📅 Last Modified: {details['last_modified']}")

        metrics = comparison.get("comparison_metrics", {})
        if metrics:
            print(f"\n🏆 Highlights:")
            if metrics.get("most_downloaded"):
                print(f"   📈 Most Downloaded: {metrics['most_downloaded']}")
            if metrics.get("most_liked"):
                print(f"   ❤️ Most Liked: {metrics['most_liked']}")

    def create_model_cards(self, output_dir: str = "model_cards") -> Dict[str, str]:
        """为所有模型创建详细的模型卡片。"""
        print("📝 Creating model cards for Questions-Gen models...")
        
        os.makedirs(output_dir, exist_ok=True)
        created_cards = {}
        
        for stage, model_name in self.qgen_models.items():
            details = self.get_model_details(stage)
            if not details:
                continue
                
            card_content = self._generate_model_card(stage, details)
            
            # Save model card
            filename = f"{stage}_model_card.md"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(card_content)
            
            created_cards[stage] = filepath
            print(f"✅ Created model card: {filepath}")

        return created_cards

    def _generate_model_card(self, stage: str, details: Dict) -> str:
        """生成模型卡片内容。"""
        stage_descriptions = {
            "stage1": {
                "title": "Questions-Gen Stage 1: Basic Mathematical Problem Generation",
                "description": "Foundation model for generating mathematical competition problems using basic pretraining on mathematical datasets.",
                "capabilities": [
                    "Generate algebra problems",
                    "Create geometry challenges", 
                    "Design calculus questions",
                    "Formulate number theory problems"
                ],
                "training": "Supervised fine-tuning on mathematical problem datasets using Unsloth optimization."
            },
            "stage2": {
                "title": "Questions-Gen Stage 2: GRPO Optimized with Variation Generation",
                "description": "Enhanced model with reinforcement learning optimization and variation generation capabilities.",
                "capabilities": [
                    "Generate high-quality competition problems",
                    "Create problem variations with different contexts",
                    "Adapt difficulty levels dynamically",
                    "Generate both original and variant problems"
                ],
                "training": "Stage 1 + GRPO (Group-based Reinforcement Policy Optimization) with variation generation training."
            },
            "final": {
                "title": "Questions-Gen Final: Complete Pipeline with Knowledge Distillation", 
                "description": "Most advanced model incorporating knowledge distillation from expert teacher models (DeepSeek-R1).",
                "capabilities": [
                    "Generate expert-level competition problems",
                    "Create sophisticated problem variations",
                    "Provide detailed mathematical reasoning",
                    "Adapt to various mathematical domains",
                    "Generate problems with high pedagogical value"
                ],
                "training": "Stage 2 + Knowledge distillation from DeepSeek-R1 teacher model for enhanced mathematical reasoning."
            }
        }

        info = stage_descriptions.get(stage, {})
        
        card = f"""# {info.get('title', f'Questions-Gen {stage.title()}')}

## Model Description

{info.get('description', 'Mathematical problem generation model.')}

**Model Name**: `{details['model_name']}`  
**Stage**: {stage}  
**Base Model**: Qwen3-14B  
**Optimization**: Unsloth  
**Precision**: FP16  

## Capabilities

"""
        
        for capability in info.get('capabilities', []):
            card += f"- {capability}\n"

        card += f"""

## Training Details

{info.get('training', 'Advanced training pipeline for mathematical problem generation.')}

### Training Configuration
- **Base Model**: Qwen3-14B (Unsloth optimized)
- **LoRA Configuration**: r=32, alpha=32, dropout=0
- **Sequence Length**: 2048 tokens
- **Precision**: Mixed FP16/BF16
- **Batch Size**: Dynamic with gradient accumulation

## Model Statistics

- **Downloads**: {details['downloads']:,}
- **Likes**: {details['likes']}
- **Model Size**: {details['model_size_gb']} GB (estimated)
- **Last Updated**: {details['last_modified']}

## Usage

### Basic Usage with Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{details['model_name']}")
model = AutoModelForCausalLM.from_pretrained(
    "{details['model_name']}",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Generate a math problem
messages = [
    {{"role": "user", "content": "Generate a challenging calculus competition problem:"}}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.8,
        do_sample=True
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response.split("assistant")[-1].strip())
```

### Usage with Unsloth

```python
from unsloth import FastLanguageModel

# Load with Unsloth optimization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{details['model_name']}",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Enable inference mode
FastLanguageModel.for_inference(model)

# Generate problem
messages = [{{"role": "user", "content": "Create a geometry olympiad problem:"}}]
inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = model.generate(
    **tokenizer(inputs, return_tensors="pt"),
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.8
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Example Outputs

### Algebra Problem Generation
**Input**: "Generate a challenging algebra competition problem:"

**Output**: 
```
Find all real solutions to the equation x⁴ - 5x² + 6 = 0.

Solution: Let y = x². Then y² - 5y + 6 = 0, which factors as (y-2)(y-3) = 0.
Therefore y = 2 or y = 3, giving us x = ±√2 or x = ±√3.
```

### Problem Variation
**Input**: "Create a variation of this problem: Find the derivative of f(x) = x³ + 2x² - 5x + 1"

**Output**:
```
Find the derivative of g(x) = 2x³ - 4x² + 7x - 3 at the point x = 1.

Solution: g'(x) = 6x² - 8x + 7, so g'(1) = 6(1) - 8(1) + 7 = 5.
```

## Model Series Comparison

| Stage | Capabilities | Best For |
|-------|-------------|----------|
| Stage 1 | Basic problem generation | Simple math problems, educational content |
| Stage 2 | + Variation generation, GRPO optimization | Competition preparation, adaptive learning |
| **Final** | + Knowledge distillation, expert reasoning | **Advanced competitions, research, expert-level content** |

## Limitations

- Optimized for English mathematical content
- May require specific prompting for best results
- Generated solutions should be verified for mathematical accuracy
- Performance varies across different mathematical domains

## Citation

```bibtex
@misc{{questions-gen-{stage},
  title={{Questions-Gen {stage.title()}: Mathematical Problem Generation}},
  author={{Xingqiang Chen}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{details['model_name']}}}
}}
```

## License

Apache 2.0

## Contact

For questions or issues, please contact the model author or create an issue in the repository.
"""

        return card

    def check_model_health(self) -> Dict[str, Dict]:
        """检查所有模型的健康状态。"""
        print("🏥 Checking health status of all Questions-Gen models...")
        
        health_report = {}
        
        for stage, model_name in self.qgen_models.items():
            print(f"\n🔍 Checking {stage} model...")
            
            health_status = {
                "accessible": False,
                "files_complete": False,
                "metadata_valid": False,
                "issues": []
            }
            
            try:
                # Check if model is accessible
                info = model_info(model_name)
                health_status["accessible"] = True
                
                # Check essential files
                if hasattr(info, 'siblings'):
                    files = [f.rfilename for f in info.siblings]
                    
                    required_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json']
                    model_files = [f for f in files if f.endswith('.safetensors')]
                    
                    missing_required = [f for f in required_files if f not in files]
                    if not missing_required and model_files:
                        health_status["files_complete"] = True
                    else:
                        if missing_required:
                            health_status["issues"].append(f"Missing files: {missing_required}")
                        if not model_files:
                            health_status["issues"].append("No model weight files found")
                
                # Check metadata
                if hasattr(info, 'pipeline_tag') and info.pipeline_tag:
                    health_status["metadata_valid"] = True
                else:
                    health_status["issues"].append("Missing or invalid metadata")
                
                print(f"✅ {stage}: Accessible")
                if health_status["files_complete"]:
                    print(f"✅ {stage}: Files complete")
                if health_status["metadata_valid"]:
                    print(f"✅ {stage}: Metadata valid")
                
                if health_status["issues"]:
                    for issue in health_status["issues"]:
                        print(f"⚠️ {stage}: {issue}")
                        
            except Exception as e:
                health_status["issues"].append(f"Access error: {str(e)}")
                print(f"❌ {stage}: {e}")
            
            health_report[stage] = health_status

        return health_report

    def generate_hf_integration_guide(self) -> str:
        """生成HuggingFace集成指南。"""
        guide = """# Questions-Gen HuggingFace Integration Guide

## Available Models

The Questions-Gen model series consists of three progressively enhanced models:

"""
        
        for stage, model_name in self.qgen_models.items():
            details = self.get_model_details(stage)
            if details:
                guide += f"### {stage.title()} Model\n"
                guide += f"- **Model**: `{model_name}`\n"
                guide += f"- **Size**: {details.get('model_size_gb', 'Unknown')} GB\n"
                guide += f"- **Downloads**: {details.get('downloads', 0):,}\n\n"

        guide += """
## Quick Start

### Installation
```bash
pip install transformers torch unsloth
```

### Basic Usage
```python
from questions_gen import ModelValidator

# Initialize validator
validator = ModelValidator()

# Load and test a model
results = validator.validate_single_model("xingqiang/questions-gen-qwen3-14b-final-merged-16bit")

# Compare all models
comparison = validator.compare_all_models()
```

### Batch Validation
```python
from questions_gen.validation import BatchValidator

# Initialize batch validator
batch_validator = BatchValidator()

# Run comprehensive validation
results = batch_validator.comparative_batch_validation(
    category="all",  # Test all categories
    tests_per_category=3
)

# Export results
batch_validator.export_results_to_csv(results)
```

## Integration with Ollama

```python
from questions_gen.utils import OllamaManager

# Initialize Ollama manager
ollama = OllamaManager()

# Push all models to Ollama
results = ollama.push_all_models()

# Test models in Ollama
for stage in ["stage1", "stage2", "final"]:
    model_name = f"questions-gen-{stage}"
    test_results = ollama.test_model_in_ollama(model_name)
```

## Advanced Features

### Quality Evaluation
```python
from questions_gen.validation import QualityEvaluator

evaluator = QualityEvaluator()

# Comprehensive evaluation
question = "Find the derivative of f(x) = x³ + 2x² - 5x + 1"
evaluation = evaluator.comprehensive_evaluation(question)

print(f"Overall Score: {evaluation['overall_score']:.3f}")
print(f"Grade: {evaluation['grade']}")
```

### Model Comparison
```python
from questions_gen.utils import HuggingFaceUtils

hf_utils = HuggingFaceUtils()

# Verify all models exist
verification = hf_utils.verify_models_exist()

# Get detailed comparison
comparison = hf_utils.compare_all_models()

# Check model health
health = hf_utils.check_model_health()
```

## Best Practices

1. **Model Selection**:
   - Use `stage1` for basic educational content
   - Use `stage2` for competition preparation with variation needs
   - Use `final` for expert-level content and research

2. **Performance Optimization**:
   - Use Unsloth for faster loading and inference
   - Enable 4-bit quantization for memory efficiency
   - Batch process for multiple questions

3. **Quality Assurance**:
   - Always validate generated content
   - Use the quality evaluator for systematic assessment
   - Cross-reference with teacher model evaluations

## Troubleshooting

### Common Issues

1. **Memory Errors**: Use 4-bit quantization or smaller batch sizes
2. **Slow Loading**: Ensure you have sufficient disk space and bandwidth
3. **Generation Quality**: Try different temperature and top-p values

### Support

- Check model health status regularly
- Monitor download and performance metrics
- Report issues through the repository

## API Reference

For complete API documentation, see the package documentation or explore the source code in the repository.
"""

        return guide

    def save_integration_guide(self, filename: str = "hf_integration_guide.md") -> str:
        """保存集成指南到文件。"""
        guide_content = self.generate_hf_integration_guide()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"📚 Integration guide saved to: {filename}")
        return filename
