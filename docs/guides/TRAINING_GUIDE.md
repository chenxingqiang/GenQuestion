# Questions-Gen 竞赛题目生成模型训练指南

## 🎯 项目概述

基于Qwen3-14B的竞赛题目生成模型，实现三阶段训练策略：
1. **基础预训练** - 在竞赛题库上进行监督微调
2. **RL GRPO优化** - 使用组策略强化学习提升题目质量
3. **知识蒸馏** - 通过教师模型集群进一步优化

## 🏗️ 系统架构

```
Questions-Gen 训练系统
├── 基础预训练 (Stage 1)
│   ├── 历史竞赛题库 (50%)
│   ├── 条件变异题 (30%)  
│   └── 创新题型 (20%)
├── RL GRPO优化 (Stage 2)
│   ├── 组策略生成 (8题/组)
│   ├── 多维奖励函数
│   └── 创新约束层
└── 知识蒸馏 (Stage 3)
    ├── DeepSeek-R1 (难度预测)
    ├── 逻辑严谨性检查
    ├── 创新性评估
    └── 教育价值评分
```

## 📦 环境准备

### 硬件要求
- **GPU**: 建议8GB+ VRAM (支持Qwen3-14B 4bit)
- **内存**: 16GB+ RAM
- **存储**: 10GB+ 可用空间

### 依赖安装

```bash
# 安装Unsloth (GPU版本)
pip install unsloth[cu118]  # CUDA 11.8
# 或
pip install unsloth[cu121]  # CUDA 12.1

# 安装其他依赖
pip install torch transformers datasets
pip install scikit-learn pandas numpy
pip install accelerate bitsandbytes
```

## 🚀 快速开始

### 1. 运行演示

```bash
# 基础测试 (CPU即可)
python quick_demo.py

# 模型验证演示
python demo_model_validation.py
```

### 2. 完整训练

```bash
# 需要8GB+ GPU内存
python questions_gen_training.py
```

### 3. 自定义配置

```python
from questions_gen_training import TrainingConfig

# 修改训练参数
TrainingConfig.MAX_STEPS_STAGE1 = 200
TrainingConfig.GROUP_SIZE = 16
TrainingConfig.LEARNING_RATE = 1e-4
```

## ⚙️ 核心组件

### 1. 创新约束层 (NoveltyConstraint)
- **功能**: 防止重复题目生成
- **原理**: 基于TF-IDF向量相似度检测
- **阈值**: 相似度>0.85时施加惩罚

```python
class NoveltyConstraint(nn.Module):
    def forward(self, x, current_question=""):
        # 计算与历史题目相似度
        # 超过阈值则惩罚输出
        return x * penalty_factor if similarity > threshold else x
```

### 2. 奖励函数 (RewardCalculator)

多维度评估体系：
- **难度分析** (40%): 基于关键词和文本长度
- **创新性** (30%): 与历史题目的差异度
- **逻辑严谨性** (20%): 推理词汇密度
- **多样性** (10%): 组内题目差异

```python
reward = 0.4*difficulty + 0.3*novelty + 0.2*rigor + 0.1*diversity
```

### 3. GRPO组策略优化

```python
# 每组生成8道题目
group_questions = generate_question_group()

# 计算奖励并选择基准
baseline = median(rewards)
advantages = [r - baseline for r in rewards]

# 梯度更新
∇J(θ) = E[(R(Q) - R(baseline)) * ∇log p_θ(Q)]
```

## 📊 训练流程

### 阶段1: 基础预训练
```python
trainer.stage1_basic_training()
# - 200步监督微调
# - 混合题型训练数据
# - LoRA高效微调
```

### 阶段2: RL GRPO优化  
```python
trainer.stage2_grpo_training()
# - 100步强化学习
# - 8题组策略生成
# - 多维奖励优化
```

### 阶段3: 知识蒸馏
```python
trainer.stage3_distillation()
# - 80步教师指导
# - DeepSeek-R1评估
# - 对抗蒸馏提升
```

## 🎯 关键特性

### ✨ 创新点
1. **组策略GRPO**: 8题并行生成，对比优化
2. **创新约束**: TF-IDF相似度防重复
3. **多维奖励**: 难度+创新+严谨+多样性
4. **渐进训练**: 三阶段递进优化

### 📈 性能优化
- **4bit量化**: 节省50%显存
- **LoRA微调**: 仅更新1-10%参数  
- **梯度累积**: 模拟大batch训练
- **检查点保存**: 支持训练恢复

## 🔧 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `MAX_SEQ_LENGTH` | 2048 | 最大序列长度 |
| `BATCH_SIZE` | 2 | 批次大小 |
| `LEARNING_RATE` | 2e-4 | 学习率 |
| `GROUP_SIZE` | 8 | GRPO组大小 |
| `LORA_R` | 32 | LoRA秩 |

## 📁 输出文件

```
checkpoints/
├── stage1_basic/     # 基础预训练模型
├── stage2_grpo/      # GRPO优化模型  
└── stage3_final/     # 最终蒸馏模型

logs/                 # 训练日志
data/                 # 训练数据缓存
```

## 🧪 推理测试

```python
# 加载训练好的模型
trainer = QuestionsGenTrainer()

# 生成题目
messages = [{"role": "user", "content": "请生成一道数学分析竞赛题目"}]
response = trainer.model.generate(...)
```

## ⚠️ 常见问题

### 1. **CUDA内存不足**
```python
# 减少batch size
TrainingConfig.BATCH_SIZE = 1
TrainingConfig.GROUP_SIZE = 4
```

### 2. **网络连接问题**
```bash
# 设置HuggingFace镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 3. **依赖版本冲突**
```bash
# 使用虚拟环境
conda create -n questions-gen python=3.9
conda activate questions-gen
```

## 🔄 扩展功能

### 1. 自定义数据集
```python
def load_custom_dataset():
    # 加载你的竞赛题库
    return custom_dataframe

data_preparer.load_competition_datasets = load_custom_dataset
```

### 2. 教师模型集成
```python
def call_teacher_model(question):
    # 调用GPT-4/Claude等API
    return evaluation_score

# 在stage3_distillation中集成
```

### 3. 评估指标
```python
def evaluate_generated_questions(questions):
    # 实现自定义评估逻辑
    return metrics_dict
```

## 📚 参考资料

- [Unsloth文档](https://docs.unsloth.ai/)
- [Qwen3模型](https://huggingface.co/Qwen/Qwen3-14B)
- [GRPO论文](https://arxiv.org/abs/2402.14740)
- [LoRA微调](https://arxiv.org/abs/2106.09685)

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

Apache License 2.0 - 详见LICENSE文件
