# Questions-Gen 文档中心

这里包含了Questions-Gen项目的完整技术文档和用户指南。

## 📚 文档目录

### 🚀 快速开始
- [用户使用指南](guides/USAGE_GUIDE.md) - 完整的使用指南，包含安装、配置和基础使用
- [训练指南](guides/TRAINING_GUIDE.md) - 模型训练的详细步骤和配置说明

### 🔧 技术文档
- [训练流程详解](technical/TRAINING_DETAILS.md) - 三阶段训练的详细技术实现
- [训练优化总结](training/OPTIMIZATION_SUMMARY.md) - 性能优化和问题解决方案

### 📖 按用户类型分类

#### 👤 普通用户 (使用已训练模型)
1. [用户使用指南](guides/USAGE_GUIDE.md) - **从这里开始**
   - 模型下载和使用
   - API调用方式
   - Ollama本地部署
   - 质量评估工具

#### 🔬 研究人员 (深入了解技术)
1. [训练流程详解](technical/TRAINING_DETAILS.md)
   - 三阶段训练架构
   - GRPO强化学习原理
   - 知识蒸馏实现
   - 技术细节和代码示例

#### 🛠️ 开发者 (自定义训练)
1. [训练指南](guides/TRAINING_GUIDE.md) - **从这里开始**
   - 环境配置
   - 自定义训练流程
   - 参数调优指南
2. [训练优化总结](training/OPTIMIZATION_SUMMARY.md)
   - 性能优化技巧
   - 问题排查指南
   - 严格模式配置

## 🎯 核心特性概览

### 🌟 主要功能
- **三阶段训练**: 基础预训练 → RL GRPO优化 → 知识蒸馏
- **智能变体生成**: 创建现有问题的高质量变体
- **多维质量评估**: 全方位题目质量评估系统
- **本地部署支持**: Ollama集成，便捷的本地推理

### 🏆 已发布模型

| 训练阶段 | HuggingFace模型 | 最适用场景 |
|---------|----------------|-----------|
| **Final** | [`xingqiang/questions-gen-qwen3-14b-final-merged-16bit`](https://huggingface.co/xingqiang/questions-gen-qwen3-14b-final-merged-16bit) | 专业级竞赛题目生成 |
| Stage 2 | [`xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit`](https://huggingface.co/xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit) | 题目变体生成 |
| Stage 1 | [`xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged`](https://huggingface.co/xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged) | 基础问题生成 |

## 🔗 快速链接

### 基础使用
```bash
# 安装包
pip install questions-gen

# 验证模型
questions-gen validate --model final --tests 5

# 质量评估
questions-gen quality "Find the derivative of f(x) = x³ + 2x² - 5x + 1" --detailed
```

### Python API
```python
from questions_gen.validation import ModelValidator

validator = ModelValidator()
results = validator.validate_single_model(
    "xingqiang/questions-gen-qwen3-14b-final-merged-16bit"
)
```

## 🤝 社区和支持

- **问题反馈**: [GitHub Issues](https://github.com/xingqiang/questions-gen/issues)
- **模型下载**: [HuggingFace Models](https://huggingface.co/xingqiang)
- **讨论交流**: [GitHub Discussions](https://github.com/xingqiang/questions-gen/discussions)

## 📄 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](../LICENSE) 文件。

---

**Questions-Gen** - 通过AI驱动的问题生成推进数学教育。 🎓✨
