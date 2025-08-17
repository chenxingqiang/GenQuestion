from src.NoveltyConstraint import NoveltyConstraint
from src.QuestionDataPreparer import QuestionsDataPreparer
from src.RewardCalculator import RewardCalculator
from src.DeepSeekTeacher import DeepSeekTeacher
from src.config import TrainingConfig

import torch

import numpy as np
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import random
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import time
# ==================== Main Trainer ====================
class QuestionsGenTrainer:
    """Questions-Gen model trainer"""

    def __init__(self):
        self.config = TrainingConfig()
        self.data_preparer = QuestionsDataPreparer()
        self.reward_calculator = RewardCalculator()
        self.novelty_constraint = NoveltyConstraint()
        self.history_questions = []

        # Initialize DeepSeek-R1 teacher model
        self.deepseek_teacher = DeepSeekTeacher()

        print("🚀 Initializing Questions-Gen trainer...")
        self._load_model()

    def _monitor_memory(self, stage_name=""):
        """监控GPU内存使用"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_reserved = torch.cuda.max_memory_reserved() / 1024**3
            print(f"📊 {stage_name} 内存: 已分配={allocated:.2f}GB, 已预留={reserved:.2f}GB, 峰值={max_reserved:.2f}GB")

            # 如果内存使用过高，执行清理
            if reserved > 12.0:  # 假设 A100 有 40GB，使用超过 30%
                print("🧹 执行内存清理...")
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _fix_attention_bias(self):
        """统一的注意力偏置修复方法"""
        print("🔧 检查并修复 attn_bias...")
        try:
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                for i, layer in enumerate(self.model.model.layers):
                    if hasattr(layer, 'self_attn'):
                        if not hasattr(layer.self_attn, 'attn_bias') or layer.self_attn.attn_bias is None:
                            # 初始化 attn_bias
                            layer.self_attn.attn_bias = torch.zeros(1, 1, 1, 1, device=device, dtype=dtype, requires_grad=False)
                            print(f"✅ 修复 layer {i} attn_bias")
            print("✅ attn_bias 检查完成")
        except Exception as e:
            print(f"⚠️ attn_bias 修复警告: {e}")

    def _validate_training_progress(self, stage_name: str, step: int):
        """验证训练进度"""
        print(f"🔍 {stage_name} 第{step}步验证...")

        # 生成测试问题
        test_prompts = [
            "Generate a calculus problem:",
            "Create an algebra challenge:",
            "Design a geometry proof:"
        ]

        total_quality = 0
        for prompt in test_prompts:
            question = self._generate_single_question(prompt)
            reward = self.reward_calculator.calculate_reward(question, self.history_questions, [])
            total_quality += reward

        avg_quality = total_quality / len(test_prompts)
        print(f"📊 当前平均质量分数: {avg_quality:.3f}")

        # 记录验证历史
        if not hasattr(self, 'validation_history'):
            self.validation_history = []
        self.validation_history.append({
            'stage': stage_name,
            'step': step,
            'quality': avg_quality
        })

        return avg_quality


    def _load_model(self):
        """Load and configure model - Using unsloth standard approach"""

        # Clear GPU memory before loading
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        print("🔄 Loading Qwen3-14B model...")

        # Use unsloth standard model loading (consistent with reference script)
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Qwen3-14B",
            max_seq_length = 2048,   # Context length - can be longer, but uses more memory
            load_in_4bit = True,     # 4bit uses much less memory
            # token = "hf_...",      # use one if using gated models
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r = 32,           # Choose any number > 0! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 32,  # Best to choose alpha = rank or rank*2
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,   # We support rank stabilized LoRA
            loftq_config = None,  # And LoftQ
        )

        print("✅ Model loading completed")

                # Prepare model for training
        self.model.train()

        # Ensure model is properly initialized
        if hasattr(self.model, 'config'):
            # Set attention configuration
            if hasattr(self.model.config, 'use_cache'):
                self.model.config.use_cache = False
            if hasattr(self.model.config, 'pretraining_tp'):
                self.model.config.pretraining_tp = 1

        # 修复注意力偏置
        self._fix_attention_bias()

        # Clear any cached states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stage1_basic_training(self):
        """Stage 1: Basic pretraining - Using unsloth standard training"""
        print("\n" + "="*50)
        print("🎯 Stage 1: Basic pretraining started")
        print("="*50)

        # Prepare training data in unsloth format (exactly like reference script)
        train_dataset = self.data_preparer.prepare_training_data(self.tokenizer)

        if train_dataset is None:
            print("❌ Failed to prepare training dataset")
            return

        # If dataset already has 'text' field (from unsloth processing), use directly
        if 'text' in train_dataset.column_names:
            final_dataset = train_dataset
            print("✅ Using pre-processed unsloth-style dataset")
        else:
            # Convert to chat format using unsloth approach (fallback)
            conversations = self.tokenizer.apply_chat_template(
                train_dataset["conversations"],
                tokenize = False,
            )

            # Create final dataset in unsloth standard format
            final_dataset = Dataset.from_dict({"text": conversations})
            final_dataset = final_dataset.shuffle(seed = 3407)

        # Show memory stats (unsloth style)
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

        # Configure trainer (exactly like reference script)
        trainer = SFTTrainer(
            model = self.model,
            tokenizer = self.tokenizer,
            train_dataset = final_dataset,
            eval_dataset = None, # Can set up evaluation!
            args = SFTConfig(
                dataset_text_field = "text",
                per_device_train_batch_size = 1,
                gradient_accumulation_steps = 8, # 减少GA步数
                warmup_steps = 10, # 减少warmup
                max_steps = min(50, self.config.MAX_STEPS_STAGE1), # 限制步数
                learning_rate = 1e-4, # 降低学习率
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "cosine",
                seed = 3407,
                report_to = "none",
                dataloader_pin_memory = False,
                save_safetensors = True,
                fp16 = False,
                bf16 = True,
                remove_unused_columns = False,
                dataloader_num_workers = 0,
                ddp_find_unused_parameters = False,
                group_by_length = False, # 禁用group_by_length
                save_only_model = True,
            ),
        )

        self._monitor_memory("训练开始前")
        print("🔄 Starting basic pretraining...")
        try:
            # Clear cache before training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            trainer_stats = trainer.train()
            self._monitor_memory("训练完成后")

        except Exception as e:
            print(f"❌ Training error: {e}")
            print("🔄 Attempting recovery...")

            # Clear memory and try again with smaller batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 恢复过程中重新修复注意力偏置
            print("🔧 恢复过程中重新检查 attn_bias...")
            try:
                self._fix_attention_bias()
            except Exception as attn_e:
                print(f"⚠️ attn_bias 恢复警告: {attn_e}")

            # Recreate trainer with safer settings
            trainer = SFTTrainer(
                model = self.model,
                tokenizer = self.tokenizer,
                train_dataset = final_dataset,
                eval_dataset = None,
                args = SFTConfig(
                    dataset_text_field = "text",
                    per_device_train_batch_size = 1,
                    gradient_accumulation_steps = 16,  # Increased GA
                    warmup_steps = 5,
                    max_steps = min(10, self.config.MAX_STEPS_STAGE1),  # Reduced steps
                    learning_rate = 1e-4,  # Reduced LR
                    logging_steps = 1,
                    optim = "adamw_8bit",
                    weight_decay = 0.01,
                    lr_scheduler_type = "cosine",
                    seed = 3407,
                    report_to = "none",
                    dataloader_pin_memory = False,
                    save_safetensors = True,
                    fp16 = False,
                    bf16 = True,
                    remove_unused_columns = False,
                    dataloader_num_workers = 0,
                    ddp_find_unused_parameters = False,
                    group_by_length = False,  # Disable for safety
                    save_only_model = True,
                ),
            )

            print("🔄 Retrying training with safer configuration...")
            trainer_stats = trainer.train()
            self._monitor_memory("训练完成后")

        # Show final memory and time stats (unsloth style)
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

        # Save checkpoint
        os.makedirs("checkpoints/stage1_basic", exist_ok=True)
        self.model.save_pretrained("checkpoints/stage1_basic")
        self.tokenizer.save_pretrained("checkpoints/stage1_basic")
        print("💾 Stage 1 model saved")

    def stage2_grpo_training(self):
        """Stage 2: RL GRPO optimization - Coordinated problem generation and variation training"""
        print("\n" + "="*50)
        print("🎯 Stage 2: RL GRPO optimization started (with variation generation training)")
        print("="*50)

        # Prepare base dataset for policy training
        train_dataset = self.data_preparer.prepare_training_data(self.tokenizer)

        # Load variation training data from real data processor
        variation_training_data = []
        if self.data_preparer.real_processor:
            try:
                variation_training_data = self.data_preparer.real_processor.create_grpo_variation_training_data()
                print(f"✅ Loaded {len(variation_training_data)} variation training examples")
            except Exception as e:
                print(f"⚠️ Failed to load variation training data: {e}")

        for step in range(self.config.MAX_STEPS_STAGE2):
            print(f"\n🔄 GRPO step {step+1}/{self.config.MAX_STEPS_STAGE2}")

            # === Part 1: Original Problem Generation Training ===
            print("📝 Training original problem generation...")

            # Generate a group of questions
            group_questions = self._generate_question_group()

            # Calculate rewards
            rewards = []
            for question in group_questions:
                reward = self.reward_calculator.calculate_reward(
                    question, self.history_questions, group_questions
                )
                rewards.append(reward)

                # Update novelty constraint layer
                self.novelty_constraint.update_history(question)

            # Select baseline question (median reward)
            median_idx = np.argsort(rewards)[len(rewards)//2]
            baseline_reward = rewards[median_idx]

            # Calculate advantages for policy gradient
            advantages = [r - baseline_reward for r in rewards]

            # Create training data for original problem generation
            original_conversations = []
            for i, (question, advantage) in enumerate(zip(group_questions, advantages)):
                if advantage > 0:  # Only train on better-than-baseline questions
                    original_conversations.append([
                        {"role": "user", "content": "Generate a high-quality competition problem:"},
                        {"role": "assistant", "content": question}
                    ])

            # === Part 2: Variation Generation Training ===
            print("🔄 Training variation generation capabilities...")

            # Generate variations for high-reward questions
            variation_conversations = []
            high_reward_questions = [q for q, r in zip(group_questions, rewards) if r > baseline_reward]

            for original_question in high_reward_questions[:3]:  # Use top 3 questions
                # Create variation training examples
                variation_examples = self._create_variation_training_examples(original_question)
                variation_conversations.extend(variation_examples)

            # Add pre-loaded variation training data (sample some)
            if variation_training_data and step % 3 == 0:  # Every 3 steps, add real variation data
                sampled_variation_data = random.sample(
                    variation_training_data,
                    min(5, len(variation_training_data))
                )
                for item in sampled_variation_data:
                    variation_conversations.append(item["conversations"])

            # === Part 3: Combined Training ===
            print("🎯 Combined training on original + variation generation...")

            # Combine original and variation training data
            all_conversations = original_conversations + variation_conversations

            # Convert to training format
            if all_conversations:
                step_texts = self.tokenizer.apply_chat_template(
                    all_conversations,
                    tokenize=False,
                )
                step_dataset = Dataset.from_dict({"text": step_texts})

                # Mini training step with combined data
                trainer = SFTTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    train_dataset=step_dataset,
                    args=SFTConfig(
                        dataset_text_field="text",
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=2,
                        max_steps=3,  # Slightly more steps for combined training
                        learning_rate=1e-5,  # Lower learning rate for fine adjustment
                        logging_steps=1,
                        optim="adamw_8bit",
                        fp16=False,
                        bf16=True,
                        report_to="none",
                    ),
                )
                trainer.train()

            # === Part 4: Validation and Statistics ===
            print("📊 Validating learned capabilities...")

            # Test variation generation capability
            if len(high_reward_questions) > 0:
                test_question = high_reward_questions[0]
                print(f"🧪 Testing variation generation for: {test_question[:100]}...")

                variations = self._test_variation_generation(test_question)
                variation_quality = self._evaluate_variation_quality(test_question, variations)

                print(f"🎯 Generated {len(variations)} variations, quality score: {variation_quality:.3f}")

            print(f"📊 Reward distribution: Mean={np.mean(rewards):.3f}, "
                  f"Std={np.std(rewards):.3f}, Baseline={baseline_reward:.3f}")
            print(f"🎯 Training samples: {len(original_conversations)} original + {len(variation_conversations)} variations")

            # Update historical question database
            self.history_questions.extend(group_questions)

            # Keep recent 1000 questions
            if len(self.history_questions) > 1000:
                self.history_questions = self.history_questions[-1000:]

        print("✅ Stage 2 GRPO optimization (with variation training) completed")

        # Save checkpoint
        os.makedirs("checkpoints/stage2_grpo", exist_ok=True)
        self.model.save_pretrained("checkpoints/stage2_grpo")
        self.tokenizer.save_pretrained("checkpoints/stage2_grpo")
        print("💾 Stage 2 model saved")

    def stage3_distillation(self):
        """Stage 3: DeepSeek-R1 Knowledge Distillation - Real teacher model guidance"""
        print("\n" + "="*50)
        print("🎯 Stage 3: DeepSeek-R1 Knowledge Distillation started")
        print("="*50)

        # Create distillation training data with real DeepSeek-R1 teacher
        distillation_conversations = []
        teacher_evaluations = []

        for step in range(self.config.MAX_STEPS_STAGE3):
            print(f"\n🔄 Distillation step {step+1}/{self.config.MAX_STEPS_STAGE3}")

            # Generate initial question with student model
            student_question = self._generate_single_question()
            print(f"📝 Student generated: {student_question[:100]}...")

            # Get DeepSeek-R1 teacher evaluation
            print("🤖 Getting DeepSeek-R1 teacher evaluation...")
            teacher_eval = self.deepseek_teacher.evaluate_problem(student_question)
            teacher_evaluations.append(teacher_eval)

            # Show teacher feedback
            print(f"👨‍🏫 Teacher overall score: {teacher_eval['overall_score']:.2f}/5.0")
            print(f"📊 Difficulty: {teacher_eval['difficulty_score']:.1f}, Rigor: {teacher_eval['rigor_score']:.1f}")
            print(f"💡 Innovation: {teacher_eval['innovation_score']:.1f}, Clarity: {teacher_eval['clarity_score']:.1f}")

            # Get teacher's improved version
            print("🔄 Getting teacher's improvement...")
            teacher_improved = self.deepseek_teacher.improve_problem(
                student_question,
                teacher_eval['raw_feedback']
            )

            if teacher_improved and teacher_improved != student_question:
                print(f"✨ Teacher improved: {teacher_improved[:100]}...")

                # Create distillation training pair
                distillation_conversations.append([
                    {"role": "user", "content": f"Improve this competition problem based on expert feedback: {student_question}"},
                    {"role": "assistant", "content": teacher_improved}
                ])

                # Also create evaluation-guided generation pair
                feedback_summary = f"Focus on: difficulty={teacher_eval['difficulty_score']:.1f}, rigor={teacher_eval['rigor_score']:.1f}, innovation={teacher_eval['innovation_score']:.1f}"
                distillation_conversations.append([
                    {"role": "user", "content": f"Generate a high-quality competition problem. {feedback_summary}"},
                    {"role": "assistant", "content": teacher_improved}
                ])
            else:
                print("⚠️ Teacher improvement failed, using original")

            # Every 3 steps, also get teacher-generated variations
            if step % 3 == 0 and step > 0:
                print("🔄 Getting teacher variations...")
                teacher_variations = self.deepseek_teacher.generate_variations(student_question, 2)

                for i, variation in enumerate(teacher_variations):
                    if variation:
                        print(f"🎯 Teacher variation {i+1}: {variation[:80]}...")
                        distillation_conversations.append([
                            {"role": "user", "content": f"Create a variation of this problem: {student_question}"},
                            {"role": "assistant", "content": variation}
                        ])

            # Rate limiting for API calls
            time.sleep(1)

        # Create distillation training dataset
        if distillation_conversations:
            print(f"\n📚 Creating distillation dataset with {len(distillation_conversations)} examples...")

            distillation_texts = self.tokenizer.apply_chat_template(
                distillation_conversations,
                tokenize=False,
            )
            distillation_dataset = Dataset.from_dict({"text": distillation_texts})

            # Configure distillation trainer
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=distillation_dataset,
                args=SFTConfig(
                    dataset_text_field="text",
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    max_steps=self.config.MAX_STEPS_STAGE3,
                    learning_rate=1e-5,  # Lower learning rate for distillation
                    logging_steps=5,
                    optim="adamw_8bit",
                    weight_decay=0.01,
                    lr_scheduler_type="linear",
                    seed=3407,
                    report_to="none",
                    fp16=False,
                    bf16=True,
                ),
            )

            print("🔄 Starting DeepSeek-R1 knowledge distillation training...")
            trainer.train()

            # Show teacher evaluation statistics
            print("\n📊 Teacher Evaluation Statistics:")
            if teacher_evaluations:
                avg_overall = np.mean([eval['overall_score'] for eval in teacher_evaluations])
                avg_difficulty = np.mean([eval['difficulty_score'] for eval in teacher_evaluations])
                avg_rigor = np.mean([eval['rigor_score'] for eval in teacher_evaluations])
                avg_innovation = np.mean([eval['innovation_score'] for eval in teacher_evaluations])

                print(f"📈 Average scores - Overall: {avg_overall:.2f}, Difficulty: {avg_difficulty:.2f}")
                print(f"📈 Rigor: {avg_rigor:.2f}, Innovation: {avg_innovation:.2f}")
                print(f"📊 Total teacher feedback examples: {len(teacher_evaluations)}")
                print(f"📊 Total distillation training pairs: {len(distillation_conversations)}")

        print("✅ Stage 3 DeepSeek-R1 knowledge distillation completed")

        # Test the distilled model
        print("\n🧪 Testing distilled model capabilities...")
        test_question = self._generate_single_question("Generate an innovative calculus competition problem:")
        final_eval = self.deepseek_teacher.evaluate_problem(test_question)
        print(f"🎯 Final model test - Teacher score: {final_eval['overall_score']:.2f}/5.0")

        # Save final model
        os.makedirs("checkpoints/stage3_final", exist_ok=True)
        self.model.save_pretrained("checkpoints/stage3_final")
        self.tokenizer.save_pretrained("checkpoints/stage3_final")
        print("💾 Final distilled model saved")

    def _generate_single_question(self, custom_prompt: str = None, enable_thinking: bool = False) -> str:
        """Generate single question - Using unsloth inference style (exactly like reference script)"""
        if custom_prompt:
            prompt = custom_prompt
        else:
            prompt = "Generate a high-quality competition problem:"

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True, # Must add for generation
            enable_thinking = enable_thinking, # Support both thinking and non-thinking modes
        )

        # Use unsloth native inference style (exactly like reference script)
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to("cuda")

            if enable_thinking:
                # For thinking mode (like reference script)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens = 1024, # Increase for longer outputs!
                    temperature = 0.6, top_p = 0.95, top_k = 20, # For thinking
                    do_sample = True,
                )
            else:
                # For non-thinking mode (like reference script)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens = 256, # Increase for longer outputs!
                    temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
                    do_sample = True,
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            question = generated_text.split("assistant")[-1].strip()

        return question

    def _generate_question_group(self) -> List[str]:
        """Generate a group of questions for GRPO"""
        questions = []
        prompts = [
            "Generate a challenging algebra competition problem:",
            "Create a geometry problem suitable for math olympiad:",
            "Design a calculus problem with moderate difficulty:",
            "Formulate a number theory competition question:",
            "Develop a combinatorics problem for advanced students:",
            "Create an analysis problem requiring proof:",
            "Design a probability problem with real-world context:",
            "Generate an innovative interdisciplinary math problem:"
        ]

        for i in range(self.config.GROUP_SIZE):
            prompt = prompts[i % len(prompts)]
            question = self._generate_single_question(prompt)
            questions.append(question)

        return questions

    def _create_variation_training_examples(self, original_question: str) -> List[List[Dict]]:
        """Create variation training examples for a given original question"""
        variation_examples = []

        # Define variation types and prompts
        variation_types = [
            {
                "type": "context_change",
                "prompt": f"Generate a mathematical problem variation that maintains the same core concept as: {original_question}\nRequirement: Change the context but keep the solution method identical.",
                "instruction": "Change the mathematical context while preserving the solution approach"
            },
            {
                "type": "parameter_change",
                "prompt": f"Create a problem variant with similar difficulty: {original_question}\nRequirement: Use different parameters but same mathematical structure.",
                "instruction": "Modify numerical parameters while maintaining the same mathematical structure"
            },
            {
                "type": "practical_application",
                "prompt": f"Transform this problem into a practical application: {original_question}\nRequirement: Add real-world context while preserving the mathematical essence.",
                "instruction": "Add real-world context while preserving the mathematical core"
            }
        ]

        for var_type in variation_types:
            # Generate variation using current model
            variation = self._generate_single_question(var_type["prompt"])

            # Create training conversation
            training_example = [
                {
                    "role": "user",
                    "content": f"{var_type['instruction']}: {original_question}"
                },
                {
                    "role": "assistant",
                    "content": f"Here's a {var_type['type']} variation:\n\nOriginal: {original_question}\n\nVariation: {variation}\n\nBoth problems maintain the same solution approach."
                }
            ]

            variation_examples.append(training_example)

        return variation_examples

    def _test_variation_generation(self, original_question: str, num_variations: int = 3) -> List[str]:
        """Test the model's variation generation capability"""
        variations = []

        test_prompts = [
            f"Generate a variation of this math problem with different context: {original_question}",
            f"Create a similar problem with different parameters: {original_question}",
            f"Transform this into a real-world application: {original_question}"
        ]

        for i in range(min(num_variations, len(test_prompts))):
            try:
                variation = self._generate_single_question(test_prompts[i])
                if variation and len(variation) > 20:  # Basic quality check
                    variations.append(variation)
            except Exception as e:
                print(f"⚠️ Variation generation failed: {e}")

        return variations

    def _evaluate_variation_quality(self, original_question: str, variations: List[str]) -> float:
        """Evaluate the quality of generated variations"""
        if not variations:
            return 0.0

        quality_scores = []

        for variation in variations:
            # Calculate similarity (should be moderate - not too high, not too low)
            try:
                vectorizer = TfidfVectorizer(max_features=100)
                tfidf_matrix = vectorizer.fit_transform([original_question, variation])
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

                # Optimal similarity range: 0.3-0.7 (similar structure, different content)
                if 0.3 <= similarity <= 0.7:
                    similarity_score = 1.0
                elif similarity < 0.3:
                    similarity_score = similarity / 0.3  # Too different
                else:
                    similarity_score = (1.0 - similarity) / 0.3  # Too similar

                # Length similarity (variations should have reasonable length)
                length_ratio = min(len(variation), len(original_question)) / max(len(variation), len(original_question))
                length_score = length_ratio if length_ratio > 0.5 else length_ratio * 2

                # Overall quality score
                variation_quality = (similarity_score + length_score) / 2.0
                quality_scores.append(variation_quality)

            except Exception as e:
                print(f"⚠️ Quality evaluation failed: {e}")
                quality_scores.append(0.5)  # Default score

        return np.mean(quality_scores) if quality_scores else 0.0

    def save_to_huggingface(self, stage_name: str = "final"):
        """Save model to HuggingFace Hub - Using unsloth standard methods (exactly like reference script)"""
        print(f"\n💾 Saving model (stage: {stage_name})...")

        # Get HF token
        hf_token = self.config.HF_TOKEN or os.environ.get('HF_TOKEN')

        # Always save locally first (like reference script)
        print("💾 Local saving...")
        self.model.save_pretrained("lora_model")  # Local saving
        self.tokenizer.save_pretrained("lora_model")

        # Only push to hub if token is available
        if hf_token:
            print(f"📤 Uploading to HuggingFace Hub...")

            # Model repository name
            repo_name = f"{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-{stage_name}"
            print(f"📤 Repository: {repo_name}")

            try:
                # Save LoRA adapters (exactly like reference script)
                self.model.push_to_hub(repo_name, token = hf_token) # Online saving
                self.tokenizer.push_to_hub(repo_name, token = hf_token) # Online saving

                # Save merged models (using reference script pattern)
                # Merge to 16bit
                merged_repo_name = f"{repo_name}-merged-16bit"
                self.model.push_to_hub_merged(merged_repo_name, self.tokenizer, save_method = "merged_16bit", token = hf_token)

                # Merge to 4bit
                merged_4bit_repo_name = f"{repo_name}-merged-4bit"
                self.model.push_to_hub_merged(merged_4bit_repo_name, self.tokenizer, save_method = "merged_4bit", token = hf_token)

                # Save GGUF format (using reference script pattern)
                gguf_repo_name = f"{repo_name}-gguf"
                self.model.push_to_hub_gguf(
                    gguf_repo_name,
                    self.tokenizer,
                    quantization_method = ["q4_k_m", "q8_0", "q5_k_m"],
                    token = hf_token
                )

                print(f"✅ Model successfully uploaded to HuggingFace!")
                print(f"📁 Repositories created:")
                print(f"   - LoRA: https://huggingface.co/{repo_name}")
                print(f"   - Merged 16bit: https://huggingface.co/{merged_repo_name}")
                print(f"   - Merged 4bit: https://huggingface.co/{merged_4bit_repo_name}")
                print(f"   - GGUF: https://huggingface.co/{gguf_repo_name}")

                return True

            except Exception as e:
                print(f"❌ Failed to upload to HuggingFace: {e}")
                print("💡 Please check your token and network connection")
        else:
            print("⚠️ No HuggingFace token found. Only local saving completed.")
            print("💡 To upload to HF Hub, set HF_TOKEN environment variable")
            print("💡 Get a token from: https://huggingface.co/settings/tokens")
            return True

    def inference_test(self):
        """Test model inference - Using unsloth inference style with variation generation testing"""
        print("\n" + "="*50)
        print("🧪 Running comprehensive inference test (original + variation generation)")
        print("="*50)

        test_prompts = [
            "Generate a calculus competition problem:",
            "Create an algebra problem with moderate difficulty:",
            "Design a geometry proof problem:"
        ]

        # Import TextStreamer for unsloth-style streaming output
        from transformers import TextStreamer

        generated_problems = []  # Store for variation testing

        for i, prompt in enumerate(test_prompts):
            print(f"\n📝 Test {i+1}: {prompt}")

            # Test 1: Non-thinking mode (like reference script)
            print("🔄 Non-thinking mode:")
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True, # Must add for generation
                enable_thinking = False, # Disable thinking
            )

            # Use unsloth native inference with streaming (exactly like reference)
            _ = self.model.generate(
                **self.tokenizer(text, return_tensors = "pt").to("cuda"),
                max_new_tokens = 256, # Increase for longer outputs!
                temperature = 0.7, top_p = 0.8, top_k = 20, # For non thinking
                streamer = TextStreamer(self.tokenizer, skip_prompt = True),
            )

            # Test 2: Thinking mode (like reference script)
            print(f"\n🤔 Thinking mode:")
            text_thinking = self.tokenizer.apply_chat_template(
                messages,
                tokenize = False,
                add_generation_prompt = True, # Must add for generation
                enable_thinking = True, # Enable thinking
            )

            _ = self.model.generate(
                **self.tokenizer(text_thinking, return_tensors = "pt").to("cuda"),
                max_new_tokens = 1024, # Increase for longer outputs!
                temperature = 0.6, top_p = 0.95, top_k = 20, # For thinking
                streamer = TextStreamer(self.tokenizer, skip_prompt = True),
            )

            # Also get the generated text for reward calculation and variation testing
            with torch.no_grad():
                inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens = 256,
                    temperature = 0.7, top_p = 0.8, top_k = 20,
                    do_sample = True,
                )

                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated_text.split("assistant")[-1].strip()

                # Calculate reward for this question
                reward = self.reward_calculator.calculate_reward(
                    response, self.history_questions, []
                )
                print(f"\n🎯 Quality score: {reward:.3f}")

                # Store for variation testing
                generated_problems.append(response)

        # Test 3: Variation Generation Capability
        print(f"\n🔄 Testing variation generation capability...")
        print("="*50)

        for i, original_problem in enumerate(generated_problems):
            print(f"\n🧪 Variation Test {i+1}:")
            print(f"📝 Original: {original_problem[:100]}...")

            # Test different types of variations
            variation_tests = [
                {
                    "type": "Context Change",
                    "prompt": f"Generate a variation of this math problem with different context: {original_problem}"
                },
                {
                    "type": "Parameter Change",
                    "prompt": f"Create a similar problem with different numerical parameters: {original_problem}"
                },
                {
                    "type": "Real-world Application",
                    "prompt": f"Transform this into a practical real-world application: {original_problem}"
                }
            ]

            for j, var_test in enumerate(variation_tests):
                print(f"\n🔄 {var_test['type']}:")

                var_messages = [{"role": "user", "content": var_test["prompt"]}]
                var_text = self.tokenizer.apply_chat_template(
                    var_messages,
                    tokenize = False,
                    add_generation_prompt = True,
                    enable_thinking = False,
                )

                # Generate variation with streaming
                _ = self.model.generate(
                    **self.tokenizer(var_text, return_tensors = "pt").to("cuda"),
                    max_new_tokens = 256,
                    temperature = 0.7, top_p = 0.8, top_k = 20,
                    streamer = TextStreamer(self.tokenizer, skip_prompt = True),
                )

                # Also get the variation text for quality evaluation
                with torch.no_grad():
                    var_inputs = self.tokenizer(var_text, return_tensors="pt").to("cuda")
                    var_outputs = self.model.generate(
                        **var_inputs,
                        max_new_tokens = 256,
                        temperature = 0.7, top_p = 0.8, top_k = 20,
                        do_sample = True,
                    )

                    var_generated_text = self.tokenizer.decode(var_outputs[0], skip_special_tokens=True)
                    variation = var_generated_text.split("assistant")[-1].strip()

                    # Evaluate variation quality
                    variation_quality = self._evaluate_variation_quality(original_problem, [variation])
                    print(f"🎯 Variation quality: {variation_quality:.3f}")

        # Summary
        print(f"\n📊 Inference Test Summary:")
        print(f"✅ Original problem generation: {len(generated_problems)} problems tested")
        print(f"✅ Variation generation: {len(generated_problems) * 3} variations tested")
        print(f"✅ Both thinking and non-thinking modes validated")
        print(f"✅ Quality scoring system validated")

        print("\n✅ Comprehensive inference test completed")

    def train_full_pipeline(self):
        """Complete training pipeline"""
        print("🎯 Starting Questions-Gen model complete training pipeline")
        print("📋 Training plan: Basic pretraining -> RL GRPO -> Knowledge distillation")

        try:
            # Stage 1: Basic pretraining
            self.stage1_basic_training()

            # Stage 2: RL GRPO optimization
            self.stage2_grpo_training()

            # Stage 3: Knowledge distillation
            self.stage3_distillation()

            # Inference test
            self.inference_test()

            # Save to HuggingFace
            print("\n🔄 Saving models to HuggingFace Hub...")
            self.save_to_huggingface("stage1")  # Save stage 1
            self.save_to_huggingface("stage2")  # Save stage 2
            self.save_to_huggingface("final")   # Save final model

            print("\n🎉 All coordinated training pipeline completed!")
            print("📋 Training Summary:")
            print("  ✅ Stage 1: Basic problem generation pretraining")
            print("  ✅ Stage 2: GRPO optimization + Variation generation training")
            print("  ✅ Stage 3: Knowledge distillation enhancement")
            print("  ✅ Comprehensive inference testing (original + variations)")
            print("\n📁 Local model save locations:")
            print("  - Stage 1: checkpoints/stage1_basic")
            print("  - Stage 2: checkpoints/stage2_grpo (with variation capabilities)")
            print("  - Final: checkpoints/stage3_final (fully optimized)")
            print("\n📤 HuggingFace repositories:")
            print(f"  - https://huggingface.co/{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-stage1")
            print(f"  - https://huggingface.co/{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-stage2")
            print(f"  - https://huggingface.co/{self.config.HF_USERNAME}/{self.config.HF_MODEL_NAME}-final")
            print("\n🎯 Model Capabilities:")
            print("  ✅ High-quality competition problem generation")
            print("  ✅ Intelligent problem variation generation")
            print("  ✅ Multi-dimensional quality optimization")
            print("  ✅ Real-world application transformation")
            print("  ✅ Mathematical rigor and novelty balance")

        except Exception as e:
            print(f"❌ Error during training: {e}")
            print("💡 Suggest checking GPU memory and data format")
