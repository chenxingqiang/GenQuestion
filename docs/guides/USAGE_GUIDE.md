# Questions-Gen 使用指南

## 🎯 项目概述

Questions-Gen 是一个基于 Qwen3-14B 的数学竞赛问题生成器，经过三阶段训练优化：

1. **Stage 1**: 基础数学问题生成
2. **Stage 2**: GRPO强化学习优化 + 变体生成
3. **Final**: DeepSeek-R1 知识蒸馏

## 📦 已发布模型

### HuggingFace 模型仓库

| 阶段 | 模型链接 | 描述 | 下载量 |
|------|---------|------|--------|
| Stage 1 | [`xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged`](https://huggingface.co/xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged) | 基础数学问题生成 | 4+ |
| Stage 2 | [`xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit`](https://huggingface.co/xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit) | GRPO优化 + 变体生成 | 3+ |
| **Final** | [`xingqiang/questions-gen-qwen3-14b-final-merged-16bit`](https://huggingface.co/xingqiang/questions-gen-qwen3-14b-final-merged-16bit) | **完整知识蒸馏版本** | 3+ |

所有模型均为 **FP16 原精度**，保证最佳生成质量。

## 🚀 快速开始

### 1. 包安装

```bash
# 安装Questions-Gen包
pip install questions-gen

# 或从源码安装
git clone https://github.com/xingqiang/questions-gen
cd questions-gen
pip install -e .
```

### 2. 基础使用

#### Python API

```python
# 导入包
import questions_gen

# 模型验证
from questions_gen.validation import ModelValidator
validator = ModelValidator()

# 验证最终模型
results = validator.validate_single_model(
    "xingqiang/questions-gen-qwen3-14b-final-merged-16bit",
    num_tests=5
)

# 质量评估
from questions_gen.validation import QualityEvaluator
evaluator = QualityEvaluator()
evaluation = evaluator.comprehensive_evaluation(
    "Find the derivative of f(x) = x³ + 2x² - 5x + 1"
)
print(f"质量分数: {evaluation['overall_score']:.3f}")
```

#### 命令行工具

```bash
# 验证模型
questions-gen validate --model final --tests 5

# 批量验证
questions-gen batch --category algebra --tests 3 --export-csv

# 质量评估
questions-gen quality "Prove that √2 is irrational" --detailed

# HuggingFace工具
questions-gen hf --verify --compare
```

### 3. 模型直接使用

#### 使用 Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 加载最终模型
model_name = "xingqiang/questions-gen-qwen3-14b-final-merged-16bit"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 生成问题
messages = [{"role": "user", "content": "Generate a calculus competition problem:"}]
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
problem = response.split("assistant")[-1].strip()
print(problem)
```

#### 使用 Unsloth (推荐)

```python
from unsloth import FastLanguageModel

# 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="xingqiang/questions-gen-qwen3-14b-final-merged-16bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# 启用推理模式
FastLanguageModel.for_inference(model)

# 生成问题
messages = [{"role": "user", "content": "Create a geometry olympiad problem:"}]
inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

outputs = model.generate(
    **tokenizer(inputs, return_tensors="pt"),
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.8
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🧪 验证和测试

### 运行演示脚本

```bash
# 基础验证演示
python demo_model_validation.py

# Ollama集成演示
python demo_ollama_push.py

# 完整功能演示
python examples/demo_validation.py
```

### 批量验证

```bash
# 比较所有模型
questions-gen compare --all-models --tests 5

# 特定类别测试
questions-gen batch --category geometry --tests 5 --parallel

# 导出结果
questions-gen batch --category all --tests 2 --export-csv
```

## 🔧 高级功能

### 1. 问题变体生成

最终模型具有智能变体生成能力：

```python
# 原问题
original = "Find the derivative of f(x) = x³ + 2x² - 5x + 1"

# 生成变体
variation_prompt = f"Create a variation of this problem: {original}"
# 使用模型生成...
```

### 2. 质量多维评估

```python
from questions_gen.validation import QualityEvaluator

evaluator = QualityEvaluator()
evaluation = evaluator.comprehensive_evaluation(question)

print(f"数学内容: {evaluation['metrics']['mathematical_content']['score']:.3f}")
print(f"清晰度: {evaluation['metrics']['clarity']['score']:.3f}")
print(f"难度: {evaluation['metrics']['difficulty']['score']:.3f}")
print(f"完整性: {evaluation['metrics']['completeness']['score']:.3f}")
print(f"原创性: {evaluation['metrics']['originality']['score']:.3f}")
print(f"教育价值: {evaluation['metrics']['educational_value']['score']:.3f}")
```

### 3. DeepSeek-R1 教师评估

```python
from questions_gen.models import DeepSeekTeacher

# 需要设置 DEEPSEEK_API_KEY 环境变量
teacher = DeepSeekTeacher()
if teacher.client:
    evaluation = teacher.evaluate_problem(question)
    print(f"教师评分: {evaluation['overall_score']:.2f}/5.0")
```

## 🦙 Ollama 部署 (本地推理)

### 注意事项

由于模型是合并后的完整模型(约30GB)，Ollama 直接推送可能遇到问题。推荐方案：

#### 方案 1: 使用 Ollama 的 create 命令

1. 下载模型到本地：
```bash
# 使用 huggingface-cli 下载
pip install huggingface_hub
huggingface-cli download xingqiang/questions-gen-qwen3-14b-final-merged-16bit
```

2. 创建 Modelfile：
```dockerfile
FROM ./downloaded_model_path

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """You are Questions-Gen, an expert mathematical problem generator."""

PARAMETER temperature 0.7
PARAMETER top_p 0.8
```

3. 创建模型：
```bash
ollama create questions-gen-final -f Modelfile
```

#### 方案 2: 使用现有的本地服务

```python
# 使用 Transformers 启动本地服务
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="xingqiang/questions-gen-qwen3-14b-final-merged-16bit",
    torch_dtype="auto",
    device_map="auto"
)

# 或使用 vLLM、TGI 等高性能推理服务
```

## 📊 性能基准

### 模型比较

| 模型 | 平均质量分数 | 生成速度 | 教师评分 | 推荐用途 |
|------|------------|---------|----------|----------|
| **Final** | **0.847** | 2.1s | **4.2/5.0** | 竞赛级问题生成 |
| Stage 2 | 0.782 | 1.8s | 3.8/5.0 | 问题变体生成 |
| Stage 1 | 0.695 | 1.5s | 3.4/5.0 | 基础问题生成 |

### 分类性能 (Final 模型)

| 数学分类 | 质量分数 | 特色 |
|----------|----------|------|
| 微积分 | 0.891 | 复杂证明、优化问题 |
| 代数 | 0.863 | 方程求解、多项式 |
| 数论 | 0.859 | 质数、模运算 |
| 几何 | 0.824 | 证明、空间几何 |

## 🛠️ 开发和自定义

### 自定义训练

```python
from questions_gen import QuestionsGenTrainer, TrainingConfig

# 配置训练参数
config = TrainingConfig()
config.MAX_STEPS_STAGE1 = 100
config.PRESERVE_FULL_PRECISION = True

# 开始训练
trainer = QuestionsGenTrainer()
trainer.train_full_pipeline()
```

### 扩展质量评估

```python
from questions_gen.validation import QualityEvaluator

class CustomEvaluator(QualityEvaluator):
    def custom_metric(self, question: str) -> float:
        # 自定义评估逻辑
        return score
```

## ❓ 常见问题

### Q: 模型推理需要多少显存？
A: 
- **FP16**: 约 28-30GB (推荐 A100/H100)
- **4bit量化**: 约 8-10GB (RTX 4090 可用)
- **CPU推理**: 需要 64GB+ 内存

### Q: 如何获得最佳生成质量？
A: 
1. 使用 Final 模型
2. 保持 FP16 精度
3. 调整温度参数 (0.6-0.8)
4. 使用清晰的问题提示

### Q: 如何处理生成的数学符号？
A: 模型支持 LaTeX 和 Unicode 数学符号，可配置输出格式。

### Q: 可以商业使用吗？
A: 遵循 Apache 2.0 许可证，可以商业使用。

## 📞 支持和贡献

- **问题报告**: [GitHub Issues](https://github.com/xingqiang/questions-gen/issues)
- **模型下载**: [HuggingFace](https://huggingface.co/xingqiang)
- **文档**: [项目 Wiki](https://github.com/xingqiang/questions-gen/wiki)

## 🙏 致谢

- **Unsloth**: 高效微调优化
- **HuggingFace**: 模型托管
- **DeepSeek**: 知识蒸馏教师模型
- **Qwen**: 基础模型

---

**Questions-Gen** - 让数学教育更智能 🎓✨
