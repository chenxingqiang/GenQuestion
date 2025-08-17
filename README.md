# Questions-Gen: AI-Powered Mathematical Competition Problem Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace Models](https://img.shields.io/badge/🤗%20HuggingFace-Models-yellow)](https://huggingface.co/xingqiang)
[![PyPI version](https://img.shields.io/badge/PyPI-v0.1.0-orange.svg)](https://pypi.org/project/questions-gen/)

**Questions-Gen** is a professional mathematical competition problem generation system based on Qwen3-14B, implementing a three-stage training strategy: Basic Pretraining → RL GRPO Optimization → Knowledge Distillation, specifically designed for generating high-quality mathematical competition problems.

## 🌟 Key Features

- **🎯 Three-Stage Training Pipeline**: Basic Pretraining → RL GRPO Optimization → Knowledge Distillation
- **🔄 Intelligent Problem Variation Generation**: Create smart variations of existing problems
- **📊 Multi-Dimensional Quality Assessment**: Comprehensive problem quality evaluation system
- **🤖 Teacher Model Integration**: Knowledge distillation from DeepSeek-R1
- **🚀 Ollama Integration**: Convenient local inference deployment
- **📈 Batch Validation Tools**: Large-scale model comparison testing
- **💎 Full Precision Models**: Original FP16 precision without quantization loss
- **⌨️ Professional CLI Tools**: Advanced command-line interface

## 🏆 Available Models

| Training Stage | HuggingFace Model | Description | Downloads |
|---------------|-------------------|-------------|-----------|
| **Stage 1** | [`xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged`](https://huggingface.co/xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged) | Basic mathematical problem generation | 4+ |
| **Stage 2** | [`xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit`](https://huggingface.co/xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit) | GRPO optimization + variation generation | 3+ |
| **Final** | [`xingqiang/questions-gen-qwen3-14b-final-merged-16bit`](https://huggingface.co/xingqiang/questions-gen-qwen3-14b-final-merged-16bit) | Complete knowledge distillation version | 3+ |

## 🚀 Quick Start

### Installation

```bash
# Install from PyPI
pip install questions-gen

# Install from source (development version)
git clone https://github.com/xingqiang/questions-gen.git
cd questions-gen
pip install -e .
```

### Basic Usage

#### Quick Demonstrations

```bash
# Run complete functionality demo
python examples/quick_demo.py

# Model validation demo
python examples/demo_model_validation.py

# Ollama deployment demo
python examples/demo_ollama_push.py
```

#### Command Line Interface

```bash
# Validate final model
questions-gen validate --model final --tests 5

# Batch validation across categories
questions-gen batch --category algebra --tests 3 --export-csv

# Quality assessment
questions-gen quality "Find the derivative of f(x) = x³ + 2x² - 5x + 1" --detailed

# Ollama integration
questions-gen ollama --push-all
questions-gen ollama --test questions-gen-final

# HuggingFace tools
questions-gen hf --verify --compare
```

#### Model Training

```bash
# Custom training (requires GPU)
python scripts/questions_gen_training.py
```

#### Deployment Tools

```bash
# Quick import to Ollama
python tools/ollama_import.py

# Complete download and conversion
python tools/download_and_convert.py
```

#### Python API

```python
from questions_gen import QuestionsGenTrainer
from questions_gen.validation import ModelValidator, BatchValidator, QualityEvaluator
from questions_gen.utils import OllamaManager, HuggingFaceUtils

# Model validation
validator = ModelValidator()
results = validator.validate_single_model(
    "xingqiang/questions-gen-qwen3-14b-final-merged-16bit",
    num_tests=5
)

# Batch validation
batch_validator = BatchValidator()
batch_results = batch_validator.comparative_batch_validation(
    category="calculus",
    tests_per_category=5
)

# Quality evaluation
evaluator = QualityEvaluator()
evaluation = evaluator.comprehensive_evaluation(
    "Prove that the square root of 2 is irrational."
)
print(f"Quality Score: {evaluation['overall_score']:.3f}")

# Ollama integration
ollama = OllamaManager()
ollama.push_all_models()
```

## 📊 Performance Results

### Model Comparison

Based on comprehensive mathematical domain testing:

| Model | Avg Quality Score | Generation Speed | Teacher Rating | Best Use Case |
|-------|------------------|------------------|----------------|---------------|
| **Final** | **0.847** | 2.1s | **4.2/5.0** | Professional competitions |
| Stage 2 | 0.782 | 1.8s | 3.8/5.0 | Problem variations |
| Stage 1 | 0.695 | 1.5s | 3.4/5.0 | Basic problem generation |

### Category Performance

| Mathematical Domain | Final Model | Stage 2 | Stage 1 |
|-------------------|-------------|---------|---------|
| Algebra | 0.863 | 0.798 | 0.712 |
| Calculus | 0.891 | 0.815 | 0.698 |
| Geometry | 0.824 | 0.763 | 0.681 |
| Number Theory | 0.859 | 0.785 | 0.704 |

## 🔧 Advanced Features

### Custom Model Training

```python
from questions_gen import QuestionsGenTrainer, TrainingConfig

# Configure training parameters
config = TrainingConfig()
config.MAX_STEPS_STAGE1 = 100
config.PRESERVE_FULL_PRECISION = True

# Complete three-stage training
trainer = QuestionsGenTrainer()
trainer.train_full_pipeline()
```

### Quality Assessment System

```python
from questions_gen.validation import QualityEvaluator

evaluator = QualityEvaluator()

# Comprehensive evaluation
question = "Find all solutions to x⁴ - 5x² + 6 = 0"
evaluation = evaluator.comprehensive_evaluation(question)

print(f"Overall Score: {evaluation['overall_score']:.3f}")
print(f"Grade: {evaluation['grade']}")
print(f"Recommendations: {evaluation['recommendations']}")
```

### Batch Processing

```python
from questions_gen.validation import BatchValidator

batch_validator = BatchValidator()

# Multi-model comparison testing
results = batch_validator.comparative_batch_validation(
    models=[
        "xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit",
        "xingqiang/questions-gen-qwen3-14b-final-merged-16bit"
    ],
    category="all",
    tests_per_category=3
)

# Export results
batch_validator.export_results_to_csv(results)
```

## 🐳 Ollama Local Deployment

Convenient local inference deployment:

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Push Questions-Gen models
questions-gen ollama --push-all

# Use the model
ollama run questions-gen-final "Generate a calculus competition problem:"
```

### API Usage

```python
import requests

def generate_problem(prompt):
    response = requests.post('http://localhost:11434/api/generate',
                           json={
                               'model': 'questions-gen-final',
                               'prompt': prompt,
                               'stream': False
                           })
    return response.json()['response']

problem = generate_problem("Create a number theory competition problem:")
print(problem)
```

## 🧪 Testing and Validation

### Comprehensive Testing

```bash
# Model validation
questions-gen validate --model all --tests 5 --save

# Category-specific testing
questions-gen batch --category geometry --tests 5 --parallel

# Quality evaluation
questions-gen quality "Prove that √2 is irrational" --detailed
```

### Model Comparison

```bash
# Compare all models
questions-gen compare --all-models --tests 5

# HuggingFace status check
questions-gen hf --compare --health
```

## 📈 Evaluation Dimensions

### Quality Assessment Metrics

- **Mathematical Content**: Concept diversity and complexity
- **Clarity**: Problem statement clarity and structure
- **Difficulty**: Appropriate challenge level
- **Completeness**: Problem setup and constraints
- **Originality**: Innovation and creativity
- **Educational Value**: Learning objectives and pedagogy

### Validation Categories

- **Algebra**: Equations, polynomials, abstract algebra
- **Geometry**: Euclidean, coordinate, solid geometry
- **Calculus**: Derivatives, integrals, optimization
- **Number Theory**: Primes, modular arithmetic, Diophantine equations
- **Combinatorics**: Counting, permutations, graph theory
- **Analysis**: Real analysis, sequences, convergence

## 🛠️ Development Guide

### Project Structure

```
questions-gen/
├── questions_gen/          # Main package code
│   ├── cli/               # Command-line interface
│   ├── core/              # Core functionality
│   ├── data/              # Data processing
│   ├── models/            # Model components
│   ├── utils/             # Utility functions
│   └── validation/        # Validation system
├── docs/                  # Complete documentation
│   ├── guides/           # User guides
│   ├── technical/        # Technical documentation
│   └── training/         # Training-related docs
├── examples/             # Demo scripts
├── scripts/              # Training scripts
├── tools/                # Utility scripts
└── tests/                # Test files
```

### Development Environment Setup

```bash
git clone https://github.com/xingqiang/questions-gen.git
cd questions-gen
pip install -e ".[dev]"
```

### Running Tests

```bash
pytest tests/ -v --cov=questions_gen
```

### Code Formatting

```bash
black questions_gen/
isort questions_gen/
flake8 questions_gen/
```

## 📚 System Architecture

### Three-Stage Training Pipeline

```
Questions-Gen Training System
├── Basic Pretraining (Stage 1)
│   ├── Historical competition problems (50%)
│   ├── Conditional variations (30%)  
│   └── Innovative problem types (20%)
├── RL GRPO Optimization (Stage 2)
│   ├── Group policy generation (8 problems/group)
│   ├── Multi-dimensional reward function
│   └── Novelty constraint layer
└── Knowledge Distillation (Stage 3)
    ├── DeepSeek-R1 (difficulty prediction)
    ├── Logic rigor checking
    ├── Innovation assessment
    └── Educational value scoring
```

### Reward Function System

```python
reward = 0.4 * difficulty + 0.3 * novelty + 0.2 * rigor + 0.1 * diversity
```

- **Difficulty Analysis** (40%): Based on keywords and text complexity
- **Innovation** (30%): Difference from historical problems
- **Logic Rigor** (20%): Reasoning vocabulary density
- **Diversity** (10%): Within-group problem variance

## 📖 Documentation

- **[Complete Documentation](docs/README.md)**: Documentation center entrance
- **[User Guide](docs/guides/USAGE_GUIDE.md)**: Complete user manual
- **[Training Guide](docs/guides/TRAINING_GUIDE.md)**: Custom model training
- **[Technical Documentation](docs/technical/TRAINING_DETAILS.md)**: In-depth technical implementation
- **[API Reference](questions_gen/)**: See docstrings in source code
- **[Example Code](examples/)**: Demo scripts and usage examples

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Unsloth**: Efficient fine-tuning optimization
- **HuggingFace**: Model hosting and transformers library
- **DeepSeek**: Teacher model in knowledge distillation
- **Qwen Team**: Base Qwen3-14B model

## 📞 Support

- **Issue Reports**: [GitHub Issues](https://github.com/xingqiang/questions-gen/issues)
- **Model Downloads**: [HuggingFace Models](https://huggingface.co/xingqiang)
- **Discussions**: [GitHub Discussions](https://github.com/xingqiang/questions-gen/discussions)

## 🔗 Related Projects

- [Unsloth](https://github.com/unslothai/unsloth) - Fast LLM fine-tuning
- [Transformers](https://github.com/huggingface/transformers) - State-of-the-art machine learning
- [Ollama](https://github.com/ollama/ollama) - Local LLM deployment

---

**Questions-Gen** - Advancing mathematical education through AI-powered problem generation. 🎓✨