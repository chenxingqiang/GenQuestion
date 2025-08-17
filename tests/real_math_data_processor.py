# -*- coding: utf-8 -*-
"""Enhanced Real Math Data Processor - Questions-Gen Optimized

Enhanced math data processor specifically designed for Questions-Gen model training objectives.

Key Optimizations:
1. Competition-level problem filtering and classification
2. Problem quality assessment and scoring
3. Difficulty progression and knowledge point mapping  
4. GRPO reward-compatible data structures
5. DeepSeek-R1 teacher model integration support
6. Coordinated variation training data generation
"""

import pandas as pd
import numpy as np
import random
import re
from datasets import Dataset, load_dataset
from typing import List, Dict, Tuple, Optional
import json
import math
from sklearn.feature_extraction.text import TfidfVectorizer

# 安全导入 unsloth（Colab 兼容性）
try:
    from unsloth.chat_templates import standardize_sharegpt
    UNSLOTH_AVAILABLE = True
    print("✅ Unsloth库已加载")
except ImportError:
    UNSLOTH_AVAILABLE = False
    print("⚠️ Unsloth库未找到，使用备用方案")
    
    # 备用的standardize_sharegpt函数
    def standardize_sharegpt(dataset):
        """备用的标准化函数，适用于没有unsloth的环境"""
        if 'conversations' in dataset.column_names:
            return dataset
        elif 'messages' in dataset.column_names:
            # 如果是messages格式，转换为conversations
            def convert_messages_to_conversations(example):
                return {'conversations': example['messages']}
            return dataset.map(convert_messages_to_conversations)
        else:
            # 假设数据已经是正确格式
            return dataset

class EnhancedMathDataProcessor:
    """增强版数学数据处理器 - 专为Questions-Gen模型优化"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.reasoning_dataset = None
        self.non_reasoning_dataset = None
        self.reasoning_conversations = None
        self.non_reasoning_conversations = None
        self.combined_dataset = None
        
        # Questions-Gen专用增强功能
        self.competition_problems = []
        self.difficulty_categories = {
            'basic': [],
            'intermediate': [],
            'advanced': [],
            'olympiad': []
        }
        self.knowledge_mapping = {}
        self.quality_scores = {}
        
        # 质量评估工具
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
    def load_real_datasets(self):
        """加载真实数据集，增强竞赛问题筛选"""
        print("🔄 加载真实数据集（增强竞赛问题筛选）...")
        
        try:
            print("📚 加载OpenMathReasoning-mini数据集...")
            self.reasoning_dataset = load_dataset("unsloth/OpenMathReasoning-mini", split = "cot")
            print(f"✅ 推理数据集加载: {len(self.reasoning_dataset)} 条目")
            
            print("📚 加载FineTome-100k数据集...")
            self.non_reasoning_dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
            print(f"✅ 对话数据集加载: {len(self.non_reasoning_dataset)} 条目")
            
            # 增强：提取和分类竞赛级问题
            print("🔍 筛选和分类竞赛问题...")
            self._extract_competition_problems()
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"❌ 数据集加载失败: {e}. 请检查网络连接或使用HF镜像。不支持降级到模拟数据。")
    
    def _extract_competition_problems(self):
        """提取和分类竞赛级问题"""
        competition_keywords = [
            'prove', 'find all', 'determine', 'show that', 'verify',
            'olympiad', 'competition', 'contest', 'amc', 'aime', 'usamo',
            'polynomial', 'inequality', 'optimization', 'combinatorics',
            'number theory', 'geometry', 'calculus', 'linear algebra'
        ]
        
        print(f"🔄 分析 {len(self.reasoning_dataset)} 个问题的竞赛相关性...")
        
        for i, item in enumerate(self.reasoning_dataset):
            if 'problem' in item and 'generated_solution' in item:
                problem = item['problem'].lower()
                solution = item['generated_solution']
                
                # 检查竞赛级特征
                competition_score = self._calculate_competition_score(problem, solution)
                difficulty_level = self._assess_difficulty_level(problem, solution)
                knowledge_points = self._extract_detailed_knowledge_points(problem)
                
                if competition_score > 0.3:  # 竞赛相关性阈值
                    problem_data = {
                        'index': i,
                        'problem': item['problem'],
                        'solution': solution,
                        'competition_score': competition_score,
                        'difficulty_level': difficulty_level,
                        'knowledge_points': knowledge_points,
                        'quality_score': self._calculate_quality_score(item['problem'], solution)
                    }
                    
                    self.competition_problems.append(problem_data)
                    self.difficulty_categories[difficulty_level].append(problem_data)
        
        print(f"✅ 提取 {len(self.competition_problems)} 个竞赛级问题")
        for level, problems in self.difficulty_categories.items():
            print(f"  📊 {level.capitalize()}: {len(problems)} 个问题")
    
    def _calculate_competition_score(self, problem: str, solution: str) -> float:
        """计算竞赛相关性得分 (0-1)"""
        score = 0.0
        
        # 竞赛关键词
        competition_indicators = {
            'prove': 0.3, 'find all': 0.25, 'determine': 0.2, 'show that': 0.25,
            'inequality': 0.2, 'polynomial': 0.15, 'optimization': 0.2,
            'olympiad': 0.4, 'competition': 0.3, 'contest': 0.3,
            'integer': 0.15, 'prime': 0.2, 'modular': 0.15,
            'triangle': 0.1, 'circle': 0.1, 'geometry': 0.15,
            'combinatorics': 0.2, 'permutation': 0.15, 'combination': 0.15
        }
        
        text = (problem + " " + solution).lower()
        for keyword, weight in competition_indicators.items():
            if keyword in text:
                score += weight
        
        # 数学复杂性指标
        if re.search(r'x\^[3-9]|x\^\{[0-9]+\}', text):  # 高次多项式
            score += 0.15
        if len(re.findall(r'[a-z]\s*=\s*[^,\s]+', text)) > 2:  # 多变量
            score += 0.1
        if 'theorem' in text or 'lemma' in text:
            score += 0.2
        
        return min(score, 1.0)
    
    def _assess_difficulty_level(self, problem: str, solution: str) -> str:
        """评估问题难度级别"""
        text = (problem + " " + solution).lower()
        
        # 高级指标
        advanced_indicators = ['olympiad', 'contest', 'prove', 'lemma', 'theorem']
        if any(indicator in text for indicator in advanced_indicators):
            if 'olympiad' in text or 'usamo' in text or 'imo' in text:
                return 'olympiad'
            return 'advanced'
        
        # 中级指标
        intermediate_indicators = ['polynomial', 'inequality', 'optimization', 'calculus']
        if any(indicator in text for indicator in intermediate_indicators):
            return 'intermediate'
        
        # 检查解答复杂性
        solution_lines = solution.split('\n')
        if len(solution_lines) > 5:  # 多步骤解答
            return 'intermediate'
        
        return 'basic'
    
    def _extract_detailed_knowledge_points(self, problem_text: str) -> List[str]:
        """提取详细知识点（竞赛导向）"""
        knowledge_patterns = {
            'Algebra': {
                'Polynomial': ['polynomial', 'degree', 'root', 'coefficient'],
                'Equation_Solving': ['equation', 'solve', 'roots', 'quadratic'],
                'Inequality': ['inequality', 'greater', 'less', 'maximum', 'minimum'],
                'Function': ['function', 'domain', 'range', 'inverse']
            },
            'Number_Theory': {
                'Prime_Numbers': ['prime', 'composite', 'factorization'],
                'Modular_Arithmetic': ['modular', 'remainder', 'congruent', 'mod'],
                'Divisibility': ['divisible', 'gcd', 'lcm', 'factor'],
                'Integer_Properties': ['integer', 'odd', 'even', 'perfect square']
            },
            'Geometry': {
                'Triangle_Geometry': ['triangle', 'angle', 'side', 'area'],
                'Circle_Geometry': ['circle', 'radius', 'diameter', 'circumference'],
                'Coordinate_Geometry': ['coordinate', 'plane', 'distance', 'slope'],
                'Solid_Geometry': ['volume', 'surface area', 'sphere', 'cube']
            },
            'Combinatorics': {
                'Counting': ['permutation', 'combination', 'counting', 'arrange'],
                'Probability': ['probability', 'random', 'expected value', 'variance'],
                'Graph_Theory': ['graph', 'vertex', 'edge', 'tree']
            },
            'Calculus': {
                'Differentiation': ['derivative', 'differentiate', 'rate of change'],
                'Integration': ['integral', 'integrate', 'area under curve'],
                'Optimization': ['maximum', 'minimum', 'critical point', 'optimize']
            }
        }
        
        found_points = []
        problem_lower = problem_text.lower()
        
        for category, subcategories in knowledge_patterns.items():
            for subcategory, keywords in subcategories.items():
                for keyword in keywords:
                    if keyword in problem_lower:
                        found_points.append(f"{category}:{subcategory}")
                        break
        
        return found_points if found_points else ['Basic_Math:Arithmetic']
    
    def _calculate_quality_score(self, problem: str, solution: str) -> float:
        """计算问题质量分数（用于GRPO训练）"""
        score = 0.0
        
        # 清晰度分数（基于长度和结构）
        problem_length = len(problem)
        if 50 <= problem_length <= 300:  # 最佳长度范围
            score += 0.25
        elif problem_length < 50:
            score += 0.1  # 太短
        else:
            score += 0.15  # 太长
        
        # 解答完整性
        solution_length = len(solution)
        if solution_length > 100:  # 充实的解答
            score += 0.25
        
        # 数学严谨性指标
        rigor_indicators = ['therefore', 'thus', 'hence', 'proof', 'qed', 'solution:']
        rigor_count = sum(1 for indicator in rigor_indicators if indicator.lower() in solution.lower())
        score += min(rigor_count * 0.1, 0.3)
        
        # 新颖性（避免常见模板）
        common_templates = ['find x', 'solve for', 'what is']
        template_penalty = sum(0.05 for template in common_templates if template in problem.lower())
        score -= template_penalty
        
        return max(0.0, min(score, 1.0))
    
    def create_grpo_training_dataset(self, num_examples: int = 1000) -> List[Dict]:
        """创建GRPO优化的训练数据集，包含奖励信号"""
        print(f"🔄 创建GRPO训练数据集，包含 {num_examples} 个样例...")
        
        if not self.competition_problems:
            print("❌ 无竞赛问题可用。请先运行 load_real_datasets()")
            return []
        
        grpo_examples = []
        
        # 跨难度级别采样问题
        for difficulty_level in ['basic', 'intermediate', 'advanced', 'olympiad']:
            level_problems = self.difficulty_categories[difficulty_level]
            if not level_problems:
                continue
            
            # 计算每个级别的样例数量
            level_count = min(len(level_problems), num_examples // 4)
            selected_problems = random.sample(level_problems, level_count)
            
            for problem_data in selected_problems:
                # 创建问题生成样例
                generation_example = {
                    'conversations': [
                        {
                            'role': 'user',
                            'content': f"生成一个{difficulty_level}级别的竞赛数学问题，类型为{problem_data['knowledge_points'][0] if problem_data['knowledge_points'] else 'algebra'}:"
                        },
                        {
                            'role': 'assistant',
                            'content': f"{problem_data['problem']}\n\n解答: {problem_data['solution']}"
                        }
                    ],
                    'metadata': {
                        'difficulty_level': difficulty_level,
                        'knowledge_points': problem_data['knowledge_points'],
                        'quality_score': problem_data['quality_score'],
                        'competition_score': problem_data['competition_score'],
                        'reward_signals': self._generate_reward_signals(problem_data)
                    }
                }
                
                grpo_examples.append(generation_example)
                
                # 创建变分样例
                variation_examples = self._create_variation_examples(problem_data)
                grpo_examples.extend(variation_examples)
        
        print(f"✅ 创建 {len(grpo_examples)} 个GRPO训练样例")
        return grpo_examples
    
    def _generate_reward_signals(self, problem_data: Dict) -> Dict:
        """生成GRPO训练的奖励信号"""
        return {
            'difficulty_reward': self._difficulty_reward(problem_data['difficulty_level']),
            'novelty_reward': problem_data['quality_score'],
            'rigor_reward': min(len(problem_data['solution']) / 200, 1.0),
            'diversity_reward': len(problem_data['knowledge_points']) / 5
        }
    
    def _difficulty_reward(self, difficulty_level: str) -> float:
        """计算基于难度的奖励"""
        difficulty_scores = {
            'basic': 0.3,
            'intermediate': 0.6,
            'advanced': 0.8,
            'olympiad': 1.0
        }
        return difficulty_scores.get(difficulty_level, 0.5)
    
    def _create_variation_examples(self, problem_data: Dict) -> List[Dict]:
        """创建变分生成样例（用于协调训练）"""
        variations = []
        
        original_problem = problem_data['problem']
        original_solution = problem_data['solution']
        
        # 类型1：参数变分
        param_variation = self._generate_parameter_variation(original_problem)
        if param_variation != original_problem:
            variations.append({
                'conversations': [
                    {
                        'role': 'user',
                        'content': f"为此问题生成参数变分: {original_problem}"
                    },
                    {
                        'role': 'assistant',
                        'content': f"{param_variation}\n\n解答: {self._adapt_solution_for_variation(original_solution, 'parameter')}"
                    }
                ],
                'metadata': {
                    'variation_type': 'parameter',
                    'original_quality': problem_data['quality_score'],
                    'difficulty_level': problem_data['difficulty_level']
                }
            })
        
        # 类型2：语境变分
        context_variation = self._generate_context_variation(original_problem)
        if context_variation != original_problem:
            variations.append({
                'conversations': [
                    {
                        'role': 'user',
                        'content': f"为此问题生成语境变分: {original_problem}"
                    },
                    {
                        'role': 'assistant',
                        'content': f"{context_variation}\n\n解答: {self._adapt_solution_for_variation(original_solution, 'context')}"
                    }
                ],
                'metadata': {
                    'variation_type': 'context',
                    'original_quality': problem_data['quality_score'],
                    'difficulty_level': problem_data['difficulty_level']
                }
            })
        
        return variations
    
    def _generate_parameter_variation(self, original_problem: str) -> str:
        """生成基于参数的变分"""
        # 提取和修改数值参数
        numbers = re.findall(r'\b\d+\.?\d*\b', original_problem)
        varied_problem = original_problem
        
        for num in numbers[:2]:  # 修改前2个数字
            try:
                old_val = float(num)
                if old_val <= 20:
                    new_val = random.randint(1, 25)
                else:
                    new_val = int(old_val * random.uniform(0.7, 1.5))
                
                varied_problem = varied_problem.replace(num, str(new_val), 1)
            except:
                continue
        
        return varied_problem
    
    def _generate_context_variation(self, original_problem: str) -> str:
        """生成基于语境的变分"""
        # 简单语境转换
        context_mappings = {
            'triangle': 'quadrilateral',
            'circle': 'ellipse', 
            'square': 'rectangle',
            'find': 'determine',
            'calculate': 'compute',
            'prove': 'show',
            'x': 'y',
            'function f': 'function g'
        }
        
        varied_problem = original_problem
        for old_term, new_term in context_mappings.items():
            if old_term in varied_problem.lower():
                varied_problem = re.sub(
                    r'\b' + re.escape(old_term) + r'\b',
                    new_term,
                    varied_problem,
                    count=1,
                    flags=re.IGNORECASE
                )
                break
        
        return varied_problem
    
    def _adapt_solution_for_variation(self, original_solution: str, variation_type: str) -> str:
        """为变分适配解答"""
        if variation_type == 'parameter':
            return "解答方法与原问题相同，更新了数值参数。"
        elif variation_type == 'context':
            return "解答方法保持一致，适配了新的语境。"
        return original_solution[:100] + "..."
    
    def create_deepseek_integration_dataset(self) -> List[Dict]:
        """创建DeepSeek-R1教师集成数据集"""
        print("🔄 创建DeepSeek集成数据集...")
        
        integration_examples = []
        
        # 问题评估样例
        for problem_data in self.competition_problems[:50]:  # 采样子集
            integration_examples.append({
                'task_type': 'evaluation',
                'input': problem_data['problem'],
                'expected_output': {
                    'difficulty': problem_data['difficulty_level'],
                    'quality_score': problem_data['quality_score'],
                    'knowledge_points': problem_data['knowledge_points']
                }
            })
        
        # 问题改进样例
        low_quality_problems = [p for p in self.competition_problems if p['quality_score'] < 0.5]
        for problem_data in low_quality_problems[:20]:
            integration_examples.append({
                'task_type': 'improvement',
                'input': {
                    'problem': problem_data['problem'],
                    'feedback': '提高清晰度和数学严谨性'
                },
                'expected_improvement_areas': ['clarity', 'rigor', 'completeness']
            })
        
        print(f"✅ 创建 {len(integration_examples)} 个DeepSeek集成样例")
        return integration_examples
    
    def prepare_unsloth_training_dataset(self, tokenizer, chat_percentage: float = 0.25) -> Dataset:
        """增强版unsloth训练数据集准备"""
        print("🔄 准备增强版unsloth训练数据集...")
        
        # 设置tokenizer
        self.tokenizer = tokenizer
        
        # 加载和处理数据集
        self.load_real_datasets()  # 这会在失败时抛出异常
        
        # 使用增强的竞赛聚焦处理
        if not self._process_enhanced_reasoning_data():
            raise RuntimeError("❌ 推理数据处理失败")
        
        if not self.process_non_reasoning_data():
            raise RuntimeError("❌ 非推理数据处理失败")
        
        # 创建增强的组合数据集
        combined_dataset = self._create_enhanced_combined_dataset(chat_percentage)
        
        print("✅ 增强版unsloth训练数据集准备完成!")
        return combined_dataset
    
    def _process_enhanced_reasoning_data(self):
        """处理推理数据（竞赛聚焦）"""
        print("🔄 处理推理数据（竞赛增强）...")
        
        enhanced_conversations = []
        
        # 优先竞赛问题
        for problem_data in self.competition_problems:
            conversation = [
                {"role": "user", "content": f"生成一个{problem_data['difficulty_level']}级别的竞赛问题:"},
                {"role": "assistant", "content": f"{problem_data['problem']}\n\n解答: {problem_data['solution']}"}
            ]
            enhanced_conversations.append(conversation)
        
        # 添加剩余问题
        remaining_count = len(self.reasoning_dataset) - len(self.competition_problems)
        if remaining_count > 0:
            for i, item in enumerate(self.reasoning_dataset):
                if i >= len(self.competition_problems):
                    if 'problem' in item and 'generated_solution' in item:
                        conversation = [
                            {"role": "user", "content": item['problem']},
                            {"role": "assistant", "content": item['generated_solution']}
                        ]
                        enhanced_conversations.append(conversation)
        
        # 应用聊天模板
        self.reasoning_conversations = self.tokenizer.apply_chat_template(
            enhanced_conversations,
            tokenize = False,
        )
        
        print(f"✅ 处理 {len(self.reasoning_conversations)} 个增强推理对话")
        return True
    
    def _create_enhanced_combined_dataset(self, chat_percentage: float) -> Dataset:
        """创建增强的组合数据集（质量优先）"""
        print(f"🔄 创建增强组合数据集，包含{chat_percentage*100}%聊天数据...")
        
        # 采样非推理数据
        non_reasoning_subset = pd.Series(self.non_reasoning_conversations)
        non_reasoning_subset = non_reasoning_subset.sample(
            int(len(self.reasoning_conversations)*(chat_percentage/(1 - chat_percentage))),
            random_state = 2407,
        )
        
        # 基于质量的组合排序
        reasoning_series = pd.Series(self.reasoning_conversations)
        
        # 如果有竞赛问题，优先高质量样例
        if self.competition_problems:
            # 按质量分数排序，高质量样例优先
            quality_sorted_indices = sorted(
                range(min(len(self.competition_problems), len(reasoning_series))),
                key=lambda i: self.competition_problems[i]['quality_score'] if i < len(self.competition_problems) else 0,
                reverse=True
            )
            
            # 重新排序推理对话，优先高质量问题
            high_quality_conversations = [self.reasoning_conversations[i] for i in quality_sorted_indices[:len(quality_sorted_indices)//2]]
            remaining_conversations = [conv for i, conv in enumerate(self.reasoning_conversations) if i not in quality_sorted_indices[:len(quality_sorted_indices)//2]]
            
            reordered_reasoning = high_quality_conversations + remaining_conversations
            reasoning_series = pd.Series(reordered_reasoning)
        
        # 组合数据集
        data = pd.concat([reasoning_series, non_reasoning_subset])
        data.name = "text"
        
        # 创建最终数据集
        self.combined_dataset = Dataset.from_pandas(pd.DataFrame(data))
        self.combined_dataset = self.combined_dataset.shuffle(seed = 3407)
        
        print(f"✅ 增强组合数据集创建: {len(self.combined_dataset)} 总条目")
        print(f"📊 竞赛问题优先: {len(self.competition_problems) if self.competition_problems else 0}")
        
        return self.combined_dataset
    
    def process_non_reasoning_data(self):
        """处理非推理数据集 - 遵循参考脚本"""
        if self.non_reasoning_dataset is None or self.tokenizer is None:
            print("❌ 请先加载数据集并设置tokenizer")
            return None
            
        print("🔄 转换非推理数据集为对话格式...")
        
        # 使用Unsloth的standardize_sharegpt函数
        dataset = standardize_sharegpt(self.non_reasoning_dataset)
        
        self.non_reasoning_conversations = self.tokenizer.apply_chat_template(
            dataset["conversations"],
            tokenize = False,
        )
        
        print(f"✅ 处理 {len(self.non_reasoning_conversations)} 个非推理对话")
        return self.non_reasoning_conversations
    
    def get_training_statistics(self) -> Dict:
        """获取综合训练统计"""
        stats = {
            'total_problems': len(self.reasoning_dataset) if self.reasoning_dataset else 0,
            'competition_problems': len(self.competition_problems),
            'difficulty_distribution': {level: len(problems) for level, problems in self.difficulty_categories.items()},
            'average_quality_score': np.mean([p['quality_score'] for p in self.competition_problems]) if self.competition_problems else 0,
            'knowledge_point_coverage': len(set().union(*[p['knowledge_points'] for p in self.competition_problems])) if self.competition_problems else 0
        }
        
        return stats
    
    def create_grpo_variation_training_data(self) -> List[Dict]:
        """为Stage 2 GRPO创建变分训练数据"""
        print("🔄 创建GRPO变分训练数据...")
        
        # 如果还没有加载数据集，先加载
        if not self.competition_problems:
            print("📊 数据集尚未加载，开始加载真实数据集...")
            self.load_real_datasets()  # 这会在失败时抛出异常
            
        if not self.competition_problems:
            raise RuntimeError("❌ 加载数据集后仍无竞赛问题可用于GRPO训练。")
        
        # 使用现有方法创建GRPO数据集
        grpo_data = self.create_grpo_training_dataset(num_examples=500)
        
        print(f"✅ 创建 {len(grpo_data)} 个GRPO变分训练样例")
        return grpo_data
    
    # ==================== 新增：三阶段训练专用方法 ====================
    
    def create_stage1_basic_dataset(self, tokenizer) -> Dataset:
        """为Stage 1基础预训练创建专门数据集"""
        print("🔄 创建Stage 1基础预训练数据集...")
        
        # 使用高质量竞赛问题 + 平衡的对话数据
        return self.prepare_unsloth_training_dataset(tokenizer, chat_percentage=0.25)
    
    def create_stage2_grpo_dataset(self, num_examples: int = 500) -> List[Dict]:
        """为Stage 2 GRPO训练创建分组比较数据集"""
        print(f"🔄 创建Stage 2 GRPO训练数据集，包含{num_examples}个分组比较样例...")
        
        grpo_groups = []
        
        # 为每个难度级别创建问题组
        for difficulty_level in ['basic', 'intermediate', 'advanced', 'olympiad']:
            level_problems = self.difficulty_categories[difficulty_level]
            if len(level_problems) < 4:  # 需要至少4个问题形成组
                continue
            
            # 创建问题组（每组8个问题用于比较）
            group_size = 8
            num_groups = min(len(level_problems) // group_size, num_examples // 4)
            
            for group_idx in range(num_groups):
                start_idx = group_idx * group_size
                group_problems = level_problems[start_idx:start_idx + group_size]
                
                # 按质量分数排序，创建奖励梯度
                group_problems.sort(key=lambda x: x['quality_score'], reverse=True)
                
                grpo_group = {
                    'group_id': f"{difficulty_level}_group_{group_idx}",
                    'difficulty_level': difficulty_level,
                    'problems': [],
                    'rewards': [],
                    'prompt': f"生成一个{difficulty_level}级别的高质量竞赛数学问题："
                }
                
                for i, problem_data in enumerate(group_problems):
                    grpo_group['problems'].append({
                        'problem': problem_data['problem'],
                        'solution': problem_data['solution'],
                        'quality_score': problem_data['quality_score']
                    })
                    
                    # 计算GRPO奖励（质量分数 + 排名奖励）
                    rank_reward = (len(group_problems) - i) / len(group_problems)
                    total_reward = problem_data['quality_score'] * 0.7 + rank_reward * 0.3
                    grpo_group['rewards'].append(total_reward)
                
                grpo_groups.append(grpo_group)
        
        print(f"✅ 创建{len(grpo_groups)}个GRPO训练组")
        return grpo_groups
    
    def create_stage3_distillation_dataset(self, num_examples: int = 200) -> List[Dict]:
        """为Stage 3知识蒸馏创建teacher-student对比数据集"""
        print(f"🔄 创建Stage 3知识蒸馏数据集，包含{num_examples}个teacher-student样例...")
        
        distillation_examples = []
        
        # 选择最高质量的竞赛问题用于蒸馏
        high_quality_problems = [p for p in self.competition_problems if p['quality_score'] > 0.7]
        high_quality_problems.sort(key=lambda x: x['quality_score'], reverse=True)
        
        selected_problems = high_quality_problems[:num_examples]
        
        for problem_data in selected_problems:
            # 创建蒸馏样例：学生生成 -> DeepSeek教师评估/改进
            distillation_example = {
                'student_input': {
                    'prompt': f"生成一个{problem_data['difficulty_level']}级别的竞赛问题，要求：",
                    'requirements': [
                        '数学严谨性高',
                        '难度适中有挑战性', 
                        '解答步骤清晰',
                        '具有教育价值'
                    ]
                },
                'student_output': {
                    'problem': problem_data['problem'],
                    'solution': problem_data['solution']
                },
                'teacher_feedback': {
                    'evaluation_dimensions': {
                        'difficulty': self._difficulty_reward(problem_data['difficulty_level']),
                        'novelty': problem_data['quality_score'],
                        'rigor': min(len(problem_data['solution']) / 200, 1.0),
                        'diversity': len(problem_data['knowledge_points']) / 5
                    },
                    'improvement_suggestions': self._generate_improvement_suggestions(problem_data),
                    'overall_score': problem_data['quality_score']
                },
                'knowledge_points': problem_data['knowledge_points'],
                'difficulty_level': problem_data['difficulty_level']
            }
            
            distillation_examples.append(distillation_example)
        
        print(f"✅ 创建{len(distillation_examples)}个蒸馏训练样例")
        return distillation_examples
    
    def _generate_improvement_suggestions(self, problem_data: Dict) -> List[str]:
        """根据问题数据生成改进建议"""
        suggestions = []
        
        quality_score = problem_data['quality_score']
        solution_length = len(problem_data['solution'])
        
        if quality_score < 0.6:
            suggestions.append("提高问题的数学严谨性和清晰度")
        
        if solution_length < 100:
            suggestions.append("提供更详细的解答步骤和推理过程")
        
        if len(problem_data['knowledge_points']) < 2:
            suggestions.append("增加问题的综合性，融合多个知识点")
        
        if problem_data['difficulty_level'] == 'basic':
            suggestions.append("适当提升问题难度，增加挑战性")
        
        if not suggestions:
            suggestions.append("保持当前高质量水准，继续发挥创新性")
        
        return suggestions
    
    def create_coordinated_variation_dataset(self, original_problems: List[Dict], num_variations_per_problem: int = 3) -> List[Dict]:
        """创建协调变分训练数据集（支持变分生成训练）"""
        print(f"🔄 创建协调变分训练数据集...")
        
        variation_dataset = []
        
        for problem_data in original_problems:
            original_problem = problem_data['problem']
            
            # 生成多种类型的变分
            variation_types = ['parameter', 'context', 'difficulty', 'knowledge_point']
            
            for variation_type in variation_types:
                for i in range(num_variations_per_problem):
                    if variation_type == 'parameter':
                        varied_problem = self._generate_parameter_variation(original_problem)
                    elif variation_type == 'context':
                        varied_problem = self._generate_context_variation(original_problem)
                    elif variation_type == 'difficulty':
                        varied_problem = self._generate_difficulty_variation(original_problem, problem_data['difficulty_level'])
                    elif variation_type == 'knowledge_point':
                        varied_problem = self._generate_knowledge_variation(original_problem, problem_data['knowledge_points'])
                    else:
                        continue
                    
                    if varied_problem != original_problem:
                        variation_example = {
                            'original': {
                                'problem': original_problem,
                                'solution': problem_data['solution'],
                                'quality_score': problem_data['quality_score']
                            },
                            'variation': {
                                'problem': varied_problem,
                                'solution': self._adapt_solution_for_variation(problem_data['solution'], variation_type),
                                'variation_type': variation_type,
                                'variation_index': i
                            },
                            'training_prompt': f"为原始问题生成{variation_type}类型的变分版本",
                            'metadata': {
                                'difficulty_level': problem_data['difficulty_level'],
                                'knowledge_points': problem_data['knowledge_points']
                            }
                        }
                        
                        variation_dataset.append(variation_example)
        
        print(f"✅ 创建{len(variation_dataset)}个协调变分训练样例")
        return variation_dataset
    
    def _generate_difficulty_variation(self, original_problem: str, current_level: str) -> str:
        """生成难度变分"""
        # 简单的难度调整策略
        if current_level == 'basic':
            # 提升到intermediate：添加复杂性
            return original_problem.replace('find', 'prove and find').replace('calculate', 'derive and calculate')
        elif current_level == 'intermediate':
            # 提升到advanced：添加约束条件
            return original_problem + " Additionally, prove that your solution is unique."
        elif current_level == 'advanced':
            # 提升到olympiad：添加推广
            return original_problem + " Generalize this result for the n-dimensional case."
        else:
            # olympiad保持原样
            return original_problem
    
    def _generate_knowledge_variation(self, original_problem: str, knowledge_points: List[str]) -> str:
        """生成知识点变分"""
        # 简单的知识点转换
        knowledge_mappings = {
            'Algebra:Polynomial': 'Number_Theory:Prime_Numbers',
            'Geometry:Triangle_Geometry': 'Geometry:Circle_Geometry',
            'Calculus:Differentiation': 'Calculus:Integration',
            'Combinatorics:Counting': 'Combinatorics:Probability'
        }
        
        varied_problem = original_problem
        
        for kp in knowledge_points:
            if kp in knowledge_mappings:
                target_kp = knowledge_mappings[kp]
                # 简单的关键词替换
                if 'Triangle' in kp:
                    varied_problem = varied_problem.replace('triangle', 'circle').replace('angle', 'radius')
                elif 'Polynomial' in kp:
                    varied_problem = varied_problem.replace('polynomial', 'prime number').replace('degree', 'divisor')
                
                break
        
        return varied_problem
    
    def validate_dataset_quality(self, dataset_name: str, dataset: any) -> Dict:
        """验证数据集质量"""
        print(f"🔍 验证{dataset_name}数据集质量...")
        
        validation_report = {
            'dataset_name': dataset_name,
            'total_examples': 0,
            'quality_distribution': {},
            'difficulty_balance': {},
            'knowledge_coverage': set(),
            'potential_issues': []
        }
        
        if isinstance(dataset, list):
            validation_report['total_examples'] = len(dataset)
            
            # 统计质量分布
            quality_scores = []
            difficulty_counts = {}
            
            for item in dataset:
                if isinstance(item, dict):
                    # 检查GRPO数据集
                    if 'problems' in item:
                        for problem in item['problems']:
                            if 'quality_score' in problem:
                                quality_scores.append(problem['quality_score'])
                    
                    # 检查蒸馏数据集
                    elif 'teacher_feedback' in item:
                        if 'overall_score' in item['teacher_feedback']:
                            quality_scores.append(item['teacher_feedback']['overall_score'])
                        if 'difficulty_level' in item:
                            difficulty_counts[item['difficulty_level']] = difficulty_counts.get(item['difficulty_level'], 0) + 1
                        if 'knowledge_points' in item:
                            validation_report['knowledge_coverage'].update(item['knowledge_points'])
            
            if quality_scores:
                validation_report['quality_distribution'] = {
                    'mean': np.mean(quality_scores),
                    'std': np.std(quality_scores),
                    'min': np.min(quality_scores),
                    'max': np.max(quality_scores)
                }
            
            validation_report['difficulty_balance'] = difficulty_counts
        
        # 检查潜在问题
        if validation_report['total_examples'] < 50:
            validation_report['potential_issues'].append("数据集规模较小，建议增加样例数量")
        
        if len(validation_report['knowledge_coverage']) < 5:
            validation_report['potential_issues'].append("知识点覆盖不足，建议增加多样性")
        
        print(f"✅ {dataset_name}质量验证完成")
        return validation_report

# ==================== 兼容性包装器 ====================
class RealMathDataProcessor(EnhancedMathDataProcessor):
    """兼容性包装器"""
    pass

# ==================== 新增：三阶段训练专用扩展方法 ====================

def create_stage1_basic_dataset(processor: EnhancedMathDataProcessor, tokenizer) -> Dataset:
    """为Stage 1基础预训练创建专门数据集"""
    print("🔄 创建Stage 1基础预训练数据集...")
    
    # 使用高质量竞赛问题 + 平衡的对话数据
    return processor.prepare_unsloth_training_dataset(tokenizer, chat_percentage=0.25)

def create_stage2_grpo_dataset(processor: EnhancedMathDataProcessor, num_examples: int = 500) -> List[Dict]:
    """为Stage 2 GRPO训练创建分组比较数据集"""
    print(f"🔄 创建Stage 2 GRPO训练数据集，包含{num_examples}个分组比较样例...")
    
    grpo_groups = []
    
    # 为每个难度级别创建问题组
    for difficulty_level in ['basic', 'intermediate', 'advanced', 'olympiad']:
        level_problems = processor.difficulty_categories[difficulty_level]
        if len(level_problems) < 4:  # 需要至少4个问题形成组
            continue
        
        # 创建问题组（每组8个问题用于比较）
        group_size = 8
        num_groups = min(len(level_problems) // group_size, num_examples // 4)
        
        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            group_problems = level_problems[start_idx:start_idx + group_size]
            
            # 按质量分数排序，创建奖励梯度
            group_problems.sort(key=lambda x: x['quality_score'], reverse=True)
            
            grpo_group = {
                'group_id': f"{difficulty_level}_group_{group_idx}",
                'difficulty_level': difficulty_level,
                'problems': [],
                'rewards': [],
                'prompt': f"生成一个{difficulty_level}级别的高质量竞赛数学问题："
            }
            
            for i, problem_data in enumerate(group_problems):
                grpo_group['problems'].append({
                    'problem': problem_data['problem'],
                    'solution': problem_data['solution'],
                    'quality_score': problem_data['quality_score']
                })
                
                # 计算GRPO奖励（质量分数 + 排名奖励）
                rank_reward = (len(group_problems) - i) / len(group_problems)
                total_reward = problem_data['quality_score'] * 0.7 + rank_reward * 0.3
                grpo_group['rewards'].append(total_reward)
            
            grpo_groups.append(grpo_group)
    
    print(f"✅ 创建{len(grpo_groups)}个GRPO训练组")
    return grpo_groups

def create_stage3_distillation_dataset(processor: EnhancedMathDataProcessor, num_examples: int = 200) -> List[Dict]:
    """为Stage 3知识蒸馏创建teacher-student对比数据集"""
    print(f"🔄 创建Stage 3知识蒸馏数据集，包含{num_examples}个teacher-student样例...")
    
    distillation_examples = []
    
    # 选择最高质量的竞赛问题用于蒸馏
    high_quality_problems = [p for p in processor.competition_problems if p['quality_score'] > 0.7]
    high_quality_problems.sort(key=lambda x: x['quality_score'], reverse=True)
    
    selected_problems = high_quality_problems[:num_examples]
    
    for problem_data in selected_problems:
        # 创建蒸馏样例：学生生成 -> DeepSeek教师评估/改进
        distillation_example = {
            'student_input': {
                'prompt': f"生成一个{problem_data['difficulty_level']}级别的竞赛问题，要求：",
                'requirements': [
                    '数学严谨性高',
                    '难度适中有挑战性', 
                    '解答步骤清晰',
                    '具有教育价值'
                ]
            },
            'student_output': {
                'problem': problem_data['problem'],
                'solution': problem_data['solution']
            },
            'teacher_feedback': {
                'evaluation_dimensions': {
                    'difficulty': processor._difficulty_reward(problem_data['difficulty_level']),
                    'novelty': problem_data['quality_score'],
                    'rigor': min(len(problem_data['solution']) / 200, 1.0),
                    'diversity': len(problem_data['knowledge_points']) / 5
                },
                'improvement_suggestions': generate_improvement_suggestions(problem_data),
                'overall_score': problem_data['quality_score']
            },
            'knowledge_points': problem_data['knowledge_points'],
            'difficulty_level': problem_data['difficulty_level']
        }
        
        distillation_examples.append(distillation_example)
    
    print(f"✅ 创建{len(distillation_examples)}个蒸馏训练样例")
    return distillation_examples

def generate_improvement_suggestions(problem_data: Dict) -> List[str]:
    """根据问题数据生成改进建议"""
    suggestions = []
    
    quality_score = problem_data['quality_score']
    solution_length = len(problem_data['solution'])
    
    if quality_score < 0.6:
        suggestions.append("提高问题的数学严谨性和清晰度")
    
    if solution_length < 100:
        suggestions.append("提供更详细的解答步骤和推理过程")
    
    if len(problem_data['knowledge_points']) < 2:
        suggestions.append("增加问题的综合性，融合多个知识点")
    
    if problem_data['difficulty_level'] == 'basic':
        suggestions.append("适当提升问题难度，增加挑战性")
    
    if not suggestions:
        suggestions.append("保持当前高质量水准，继续发挥创新性")
    
    return suggestions

def create_coordinated_variation_dataset(processor: EnhancedMathDataProcessor, num_variations_per_problem: int = 3) -> List[Dict]:
    """创建协调变分训练数据集（支持变分生成训练）"""
    print(f"🔄 创建协调变分训练数据集...")
    
    variation_dataset = []
    
    for problem_data in processor.competition_problems[:100]:  # 使用前100个高质量问题
        original_problem = problem_data['problem']
        
        # 生成多种类型的变分
        variation_types = ['parameter', 'context']
        
        for variation_type in variation_types:
            for i in range(num_variations_per_problem):
                if variation_type == 'parameter':
                    varied_problem = processor._generate_parameter_variation(original_problem)
                elif variation_type == 'context':
                    varied_problem = processor._generate_context_variation(original_problem)
                else:
                    continue
                
                if varied_problem != original_problem:
                    variation_example = {
                        'original': {
                            'problem': original_problem,
                            'solution': problem_data['solution'],
                            'quality_score': problem_data['quality_score']
                        },
                        'variation': {
                            'problem': varied_problem,
                            'solution': processor._adapt_solution_for_variation(problem_data['solution'], variation_type),
                            'variation_type': variation_type,
                            'variation_index': i
                        },
                        'training_prompt': f"为原始问题生成{variation_type}类型的变分版本",
                        'metadata': {
                            'difficulty_level': problem_data['difficulty_level'],
                            'knowledge_points': problem_data['knowledge_points']
                        }
                    }
                    
                    variation_dataset.append(variation_example)
    
    print(f"✅ 创建{len(variation_dataset)}个协调变分训练样例")
    return variation_dataset

def validate_dataset_quality(dataset_name: str, dataset: any) -> Dict:
    """验证数据集质量"""
    print(f"🔍 验证{dataset_name}数据集质量...")
    
    validation_report = {
        'dataset_name': dataset_name,
        'total_examples': 0,
        'quality_distribution': {},
        'difficulty_balance': {},
        'knowledge_coverage': set(),
        'potential_issues': []
    }
    
    if isinstance(dataset, list):
        validation_report['total_examples'] = len(dataset)
        
        # 统计质量分布
        quality_scores = []
        difficulty_counts = {}
        
        for item in dataset:
            if isinstance(item, dict):
                # 检查GRPO数据集
                if 'problems' in item:
                    for problem in item['problems']:
                        if 'quality_score' in problem:
                            quality_scores.append(problem['quality_score'])
                
                # 检查蒸馏数据集
                elif 'teacher_feedback' in item:
                    if 'overall_score' in item['teacher_feedback']:
                        quality_scores.append(item['teacher_feedback']['overall_score'])
                    if 'difficulty_level' in item:
                        difficulty_counts[item['difficulty_level']] = difficulty_counts.get(item['difficulty_level'], 0) + 1
                    if 'knowledge_points' in item:
                        validation_report['knowledge_coverage'].update(item['knowledge_points'])
        
        if quality_scores:
            validation_report['quality_distribution'] = {
                'mean': np.mean(quality_scores),
                'std': np.std(quality_scores),
                'min': np.min(quality_scores),
                'max': np.max(quality_scores)
            }
        
        validation_report['difficulty_balance'] = difficulty_counts
    
    # 检查潜在问题
    if validation_report['total_examples'] < 50:
        validation_report['potential_issues'].append("数据集规模较小，建议增加样例数量")
    
    if len(validation_report['knowledge_coverage']) < 5:
        validation_report['potential_issues'].append("知识点覆盖不足，建议增加多样性")
    
    print(f"✅ {dataset_name}质量验证完成")
    return validation_report

# ==================== 测试函数 ====================
def test_enhanced_processor():
    """测试增强处理器功能"""
    print("🧪 测试增强数学数据处理器")
    print("=" * 50)
    
    processor = EnhancedMathDataProcessor()
    
    # 测试增强数据集加载
    if not processor.load_real_datasets():
        print("❌ 增强数据集加载失败")
        return False
    
    # 测试GRPO数据集创建
    grpo_dataset = processor.create_grpo_training_dataset(100)
    if grpo_dataset:
        print(f"✅ GRPO数据集创建: {len(grpo_dataset)} 个样例")
    
    # 测试DeepSeek集成数据集
    deepseek_dataset = processor.create_deepseek_integration_dataset()
    if deepseek_dataset:
        print(f"✅ DeepSeek集成数据集创建: {len(deepseek_dataset)} 个样例")
    
    # 测试三阶段训练专用数据集
    if processor.competition_problems:
        # 测试Stage 2 GRPO数据集
        stage2_dataset = create_stage2_grpo_dataset(processor, 100)
        print(f"✅ Stage 2 GRPO数据集: {len(stage2_dataset)} 个训练组")
        
        # 测试Stage 3蒸馏数据集
        stage3_dataset = create_stage3_distillation_dataset(processor, 50)
        print(f"✅ Stage 3蒸馏数据集: {len(stage3_dataset)} 个样例")
        
        # 测试协调变分数据集
        variation_dataset = create_coordinated_variation_dataset(processor, 2)
        print(f"✅ 协调变分数据集: {len(variation_dataset)} 个样例")
    
    # 获取训练统计
    stats = processor.get_training_statistics()
    print(f"\n📊 训练统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return True

if __name__ == "__main__":
    print("🚀 增强版真实数学数据处理器 - Questions-Gen优化")
    print("=" * 70)
    
    # 测试增强功能
    test_enhanced_processor() 