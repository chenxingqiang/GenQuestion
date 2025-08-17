# Tools

This directory contains utility tools for Questions-Gen model management and deployment.

## 🛠️ Available Tools

### Model Management Tools

- **`download_and_convert.py`** - Download HuggingFace models and convert for Ollama
  ```bash
  python tools/download_and_convert.py
  ```

- **`ollama_import.py`** - Direct import of HuggingFace models to Ollama
  ```bash
  python tools/ollama_import.py
  ```

## 📋 Tool Descriptions

### `download_and_convert.py`
Complete model download and conversion pipeline:

**Features:**
- ✅ Downloads models from HuggingFace using git-lfs
- ✅ Creates Ollama-compatible Modelfiles
- ✅ Converts models to Ollama format
- ✅ Supports batch processing of multiple models
- ✅ Automatic cleanup of temporary files

**Usage:**
```bash
python tools/download_and_convert.py

# Options:
# 1. Download final model only (recommended)
# 2. Download all models (stage1, stage2, final)
# 3. Custom selection
```

**Requirements:**
- `git-lfs` for large file downloads
- `huggingface_hub` Python package
- Ollama installed and running

### `ollama_import.py`
Streamlined direct import tool:

**Features:**
- ✅ Direct HuggingFace to Ollama import
- ✅ Automatic Modelfile generation
- ✅ Template optimization for math problems
- ✅ Quick model testing
- ✅ Simplified workflow

**Usage:**
```bash
python tools/ollama_import.py

# Options:
# 1. Import all models
# 2. Import final model only (recommended)
# 3. Custom selection
```

## 🎯 Available Models

Both tools can work with these Questions-Gen models:

| Stage | HuggingFace Repository | Ollama Name |
|-------|------------------------|-------------|
| **Stage 1** | `xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged` | `questions-gen-stage1` |
| **Stage 2** | `xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit` | `questions-gen-stage2` |
| **Final** | `xingqiang/questions-gen-qwen3-14b-final-merged-16bit` | `questions-gen-final` |

## 🔧 Prerequisites

### System Requirements
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Install git-lfs
brew install git-lfs  # macOS
sudo apt install git-lfs  # Ubuntu
```

### Python Dependencies
```bash
pip install huggingface_hub
pip install requests  # for API calls
```

## 🚀 Quick Start

### Recommended Workflow

1. **Install Ollama** (if not already installed):
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Import the final model**:
   ```bash
   python tools/ollama_import.py
   # Choose option 2 (final model only)
   ```

3. **Test the model**:
   ```bash
   ollama run questions-gen-final "Generate a calculus competition problem:"
   ```

### Alternative: Full Download Workflow

For offline use or custom modifications:
```bash
python tools/download_and_convert.py
# Choose option 1 (final model)
# Models will be downloaded and converted locally
```

## 🎯 Ollama Integration

### Generated Modelfile Template
Both tools create optimized Modelfiles:

```dockerfile
FROM {hf_model_name}

TEMPLATE """<|im_start|>system
You are a helpful assistant that generates high-quality mathematical problems.
<|im_end|>
<|im_start|>user
{{ .Prompt }}
<|im_end|>
<|im_start|>assistant
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1

SYSTEM """You are an expert mathematics problem generator..."""
```

### Usage Examples
After successful import:

```bash
# Basic generation
ollama run questions-gen-final "Generate an algebra problem:"

# Advanced prompts
ollama run questions-gen-final "Create a challenging calculus competition problem with integration by parts:"

# Variation generation
ollama run questions-gen-final "Transform this problem into a real-world application: Find the derivative of x²"
```

## 🔍 Troubleshooting

### Common Issues

1. **Ollama not found**
   ```bash
   # Install Ollama first
   curl -fsSL https://ollama.ai/install.sh | sh
   # Restart terminal after installation
   ```

2. **HuggingFace download fails**
   ```bash
   # Check git-lfs installation
   git lfs version
   
   # Install if missing
   brew install git-lfs  # macOS
   sudo apt install git-lfs  # Ubuntu
   ```

3. **Model import fails**
   ```bash
   # Check Ollama service
   ollama --version
   
   # Restart Ollama if needed
   ollama serve
   ```

4. **Permission errors**
   ```bash
   # Ensure proper permissions
   chmod +x tools/*.py
   ```

### Network Issues
For users in regions with restricted access:
```bash
# Use HuggingFace mirror
export HF_ENDPOINT=https://hf-mirror.com
python tools/ollama_import.py
```

## 📊 Performance Notes

### Model Sizes
- **Stage 1**: ~28GB (FP16 merged)
- **Stage 2**: ~28GB (FP16 merged)  
- **Final**: ~28GB (FP16 merged)

### Download Times
- **Fast internet (100Mbps)**: ~45 minutes per model
- **Standard internet (25Mbps)**: ~3 hours per model
- **Slow internet (5Mbps)**: ~15 hours per model

### Disk Space Requirements
- **Single model**: ~30GB temporary + ~28GB final
- **All models**: ~90GB temporary + ~84GB final

## 📚 Related Documentation

- [Ollama Usage Guide](../docs/guides/USAGE_GUIDE.md#ollama-integration)
- [Model Validation](../examples/README.md)
- [Training Pipeline](../scripts/README.md)
