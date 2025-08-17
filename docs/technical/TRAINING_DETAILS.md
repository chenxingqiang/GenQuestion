# Questions-Gen 训练流程详细技术文档

## 🎯 总体架构

基于Qwen3-14B的三阶段递进式训练pipeline，专门用于生成高质量数学竞赛题目。

## 🚀 **阶段一：基础预训练 (Stage 1: Basic Pretraining)**

### 目标
建立基础的数学问题生成能力

### 数据准备
1. **真实数据集加载**：
   - 使用`OpenMathReasoning-mini`数据集（19,252条数学推理数据）
   - 使用`FineTome-100k`对话数据集（100,000条对话数据）
   - 按25%比例混合推理和对话数据（25,669条总数据）

2. **数据格式转换**：
   ```python
   # 转换为unsloth标准格式
   conversations = [
       [
           {"role": "user", "content": "Generate a challenging algebra problem:"},
           {"role": "assistant", "content": "Find x: x^4 - 5x^2 + 6 = 0..."}
       ]
   ]
   ```

### 模型配置
```python
# 模型加载（解决了所有兼容性问题）
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-14B",
    max_seq_length=2048,
    load_in_4bit=True,
    trust_remote_code=True,  # 解决attention bias错误
)

# LoRA配置
model = FastLanguageModel.get_peft_model(
    model,
    r=32,                    # LoRA rank
    lora_alpha=32,          # LoRA alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"]
)
```

### 训练参数
```python
SFTConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,    # 有效batch size = 8
    max_steps=100,                    # 训练步数
    learning_rate=2e-4,               # 学习率
    fp16=False, bf16=True,           # 正确的精度设置
    optim="adamw_8bit",              # 8bit优化器
)
```

## 🎯 **阶段二：GRPO强化学习优化 (Stage 2: RL GRPO Training)**

### 目标
通过群体奖励策略优化（GRPO）提升问题质量，并训练变体生成能力

### 核心流程（每个训练步骤）

#### 1. **原始问题生成训练**
```python
# 生成问题组（8个问题）
group_questions = self._generate_question_group()

# 多维度奖励计算
for question in group_questions:
    reward = reward_calculator.calculate_reward(
        question, history_questions, group_questions
    )
    # 奖励维度：难度(40%) + 新颖性(30%) + 严谨性(20%) + 多样性(10%)
```

#### 2. **变体生成训练**
```python
# 为高奖励问题生成训练样本
variation_types = [
    "context_change",        # 改变数学背景，保持解法
    "parameter_change",      # 修改参数，保持结构
    "practical_application"  # 添加实际应用场景
]

# 创建变体训练对话
training_example = [
    {"role": "user", "content": "Change context: [original_problem]"},
    {"role": "assistant", "content": "Variation: [generated_variation]"}
]
```

#### 3. **协调训练**
```python
# 合并原始和变体训练数据
all_conversations = original_conversations + variation_conversations

# 联合训练（更低学习率精调）
trainer = SFTTrainer(
    train_dataset=step_dataset,
    args=SFTConfig(
        learning_rate=1e-5,  # 更低的学习率
        max_steps=3,         # 短步训练
        fp16=False, bf16=True
    )
)
```

### 奖励计算系统
```python
class RewardCalculator:
    def calculate_reward(self, question, history, group):
        difficulty = self.calculate_difficulty(question)      # 关键词+复杂度
        novelty = self.calculate_novelty(question, history)   # TF-IDF相似度
        rigor = self.calculate_rigor(question)               # 逻辑严谨性
        diversity = self.calculate_diversity(question, group) # 组内多样性
        
        # 加权合并 (0.4, 0.3, 0.2, 0.1)
        return weighted_combination(difficulty, novelty, rigor, diversity)
```

## 🤖 **阶段三：知识蒸馏 (Stage 3: DeepSeek-R1 Knowledge Distillation)**

### 目标
利用DeepSeek-R1作为教师模型，进行知识蒸馏优化

### 教师模型集成
```python
class DeepSeekTeacher:
    def __init__(self):
        self.client = OpenAI(
            api_key="sk-your-api-key",
            base_url="https://api.deepseek.com"
        )
    
    def evaluate_problem(self, problem):
        # 5维度评估：严谨性、难度、创新性、清晰度、教育价值
        
    def improve_problem(self, problem, feedback):
        # 基于反馈改进问题
        
    def generate_variations(self, original, num=3):
        # 生成智能变体
```

### 蒸馏训练流程
```python
for step in range(30):  # 30个蒸馏步骤
    # 1. 学生模型生成问题
    student_question = self._generate_single_question()
    
    # 2. 教师评估
    teacher_eval = deepseek_teacher.evaluate_problem(student_question)
    
    # 3. 教师改进
    teacher_improved = deepseek_teacher.improve_problem(
        student_question, teacher_eval['raw_feedback']
    )
    
    # 4. 创建训练对
    distillation_conversations.append([
        {"role": "user", "content": f"Improve: {student_question}"},
        {"role": "assistant", "content": teacher_improved}
    ])
    
    # 5. 每3步生成变体
    if step % 3 == 0:
        variations = deepseek_teacher.generate_variations(student_question)
        # 添加变体训练对...
```

## 🧪 **推理测试与验证**

### 综合测试
```python
def inference_test(self):
    # 1. 原始问题生成测试
    test_prompts = [
        "Generate a calculus competition problem:",
        "Create an algebra problem:",
        "Design a geometry proof:"
    ]
    
    # 2. 思维模式测试
    for prompt in test_prompts:
        # 非思维模式 (enable_thinking=False)
        # 思维模式 (enable_thinking=True, 更长输出)
    
    # 3. 变体生成能力测试
    for original_problem in generated_problems:
        variations = [
            "Context Change",
            "Parameter Change", 
            "Real-world Application"
        ]
        # 评估变体质量...
```

## 💾 **模型保存与部署**

### HuggingFace Hub集成
```python
def save_to_huggingface(self, stage_name):
    # 1. 本地保存
    model.save_pretrained("lora_model")
    
    # 2. 上传到HF Hub
    model.push_to_hub(f"{username}/{model_name}-{stage}")
    
    # 3. 多格式保存
    model.push_to_hub_merged(repo_name, save_method="merged_16bit")
    model.push_to_hub_merged(repo_name, save_method="merged_4bit") 
    model.push_to_hub_gguf(repo_name, quantization_method=["q4_k_m", "q8_0"])
```

## 🎯 **关键技术特性**

### 1. **新颖性约束层**
```python
class NoveltyConstraint(nn.Module):
    def forward(self, x, current_question):
        # 计算与历史问题的相似度
        # 相似度 > 0.85 时施加惩罚
        return x * penalty_factor if too_similar else x
```

### 2. **协调训练策略**
- 40%基础生成 + 30%变体生成 + 20%创新优化
- 动态奖励调整
- 历史问题数据库管理（保持最近1000个问题）

### 3. **多精度兼容**
- 解决了torch_dtype重复参数问题
- 正确设置fp16=False, bf16=True
- 添加trust_remote_code=True解决attention bias

## 📊 **训练监控与评估**

### 实时监控指标
```python
# GRPO阶段监控
print(f"📊 Reward distribution: Mean={np.mean(rewards):.3f}")
print(f"🎯 Training samples: {len(original)} original + {len(variations)} variations")

# 蒸馏阶段监控  
print(f"👨‍🏫 Teacher score: {teacher_eval['overall_score']:.2f}/5.0")
print(f"📊 Difficulty: {difficulty:.1f}, Rigor: {rigor:.1f}")
```

### 最终评估
- 问题生成质量评分
- 变体生成能力测试
- 教师模型认可度统计
- 多维度能力平衡检查

这个训练流程确保了模型既能生成高质量的原创数学竞赛题目，又具备智能变体生成能力，通过三阶段递进训练达到专业水准。
