# -*- coding: utf-8 -*-
"""Model validation and testing utilities."""

import os
import json
import time
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
from ..models.reward_calculator import RewardCalculator
from ..models.deepseek_teacher import DeepSeekTeacher


class ModelValidator:
    """Comprehensive model validation for Questions-Gen series models."""

    def __init__(self):
        self.reward_calculator = RewardCalculator()
        self.deepseek_teacher = DeepSeekTeacher()
        self.test_prompts = [
            "Generate a challenging algebra competition problem:",
            "Create a geometry problem suitable for math olympiad:",
            "Design a calculus problem with moderate difficulty:",
            "Formulate a number theory competition question:",
            "Develop a combinatorics problem for advanced students:",
            "Create a probability problem with real-world context:",
            "Generate an innovative interdisciplinary math problem:",
            "Design a proof-based analysis problem:"
        ]
        
        # 已训练的模型列表
        self.trained_models = {
            "stage1": "xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged",
            "stage2": "xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit", 
            "final": "xingqiang/questions-gen-qwen3-14b-final-merged-16bit"
        }

    def load_model_from_hf(self, model_name: str, use_unsloth: bool = True):
        """从HuggingFace加载模型。"""
        print(f"🔄 Loading model: {model_name}")
        
        if use_unsloth:
            try:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_name,
                    max_seq_length=2048,
                    load_in_4bit=True,
                )
                # Enable inference mode
                FastLanguageModel.for_inference(model)
                print(f"✅ Loaded with Unsloth: {model_name}")
                return model, tokenizer
            except Exception as e:
                print(f"⚠️ Unsloth loading failed: {e}, trying standard transformers...")
        
        # Fallback to standard transformers
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_4bit=True
            )
            print(f"✅ Loaded with transformers: {model_name}")
            return model, tokenizer
        except Exception as e:
            print(f"❌ Failed to load model {model_name}: {e}")
            return None, None

    def generate_question(self, model, tokenizer, prompt: str, 
                         enable_thinking: bool = False, 
                         max_tokens: int = 512) -> str:
        """使用模型生成问题。"""
        try:
            # Format prompt for chat
            messages = [{"role": "user", "content": prompt}]
            
            # Try unsloth-style chat template first
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
            except:
                # Fallback to simple formatting
                text = f"User: {prompt}\nAssistant:"

            # Generate response
            inputs = tokenizer(text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )

            # Decode response
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract assistant response
            if "assistant" in generated_text.lower():
                response = generated_text.split("assistant")[-1].strip()
            elif "Assistant:" in generated_text:
                response = generated_text.split("Assistant:")[-1].strip()
            else:
                response = generated_text[len(text):].strip()

            return response

        except Exception as e:
            print(f"⚠️ Generation failed: {e}")
            return "Generation failed"

    def validate_single_model(self, model_name: str, num_tests: int = 8) -> Dict:
        """验证单个模型的性能。"""
        print(f"\n🧪 Validating model: {model_name}")
        print("="*60)

        # Load model
        model, tokenizer = self.load_model_from_hf(model_name)
        if model is None:
            return {"error": f"Failed to load model {model_name}"}

        results = {
            "model_name": model_name,
            "timestamp": time.time(),
            "test_results": [],
            "statistics": {}
        }

        total_quality = 0
        generation_times = []
        teacher_scores = []

        for i, prompt in enumerate(self.test_prompts[:num_tests]):
            print(f"\n📝 Test {i+1}/{num_tests}: {prompt[:50]}...")
            
            # Generate question
            start_time = time.time()
            question = self.generate_question(model, tokenizer, prompt)
            generation_time = time.time() - start_time
            generation_times.append(generation_time)

            print(f"🤖 Generated: {question[:100]}...")
            print(f"⏱️ Generation time: {generation_time:.2f}s")

            # Calculate quality score
            quality_score = self.reward_calculator.calculate_reward(question, [], [])
            total_quality += quality_score

            # Get teacher evaluation (if available)
            teacher_eval = None
            if self.deepseek_teacher.client:
                try:
                    teacher_eval = self.deepseek_teacher.evaluate_problem(question)
                    teacher_scores.append(teacher_eval['overall_score'])
                    print(f"👨‍🏫 Teacher score: {teacher_eval['overall_score']:.2f}/5.0")
                except Exception as e:
                    print(f"⚠️ Teacher evaluation failed: {e}")

            # Test variation generation
            variation_quality = 0
            try:
                variation_prompt = f"Generate a variation of this problem: {question}"
                variation = self.generate_question(model, tokenizer, variation_prompt)
                if variation and len(variation) > 20:
                    variation_quality = self._evaluate_variation_similarity(question, variation)
                print(f"🔄 Variation quality: {variation_quality:.3f}")
            except Exception as e:
                print(f"⚠️ Variation test failed: {e}")

            # Store test result
            test_result = {
                "test_id": i + 1,
                "prompt": prompt,
                "generated_question": question,
                "quality_score": quality_score,
                "generation_time": generation_time,
                "teacher_evaluation": teacher_eval,
                "variation_quality": variation_quality
            }
            results["test_results"].append(test_result)

        # Calculate statistics
        avg_quality = total_quality / num_tests
        avg_generation_time = np.mean(generation_times)
        avg_teacher_score = np.mean(teacher_scores) if teacher_scores else None

        results["statistics"] = {
            "average_quality_score": avg_quality,
            "average_generation_time": avg_generation_time,
            "average_teacher_score": avg_teacher_score,
            "total_tests": num_tests,
            "successful_generations": len([r for r in results["test_results"] if len(r["generated_question"]) > 20])
        }

        print(f"\n📊 Model Validation Summary:")
        print(f"   Average Quality Score: {avg_quality:.3f}")
        print(f"   Average Generation Time: {avg_generation_time:.2f}s")
        if avg_teacher_score:
            print(f"   Average Teacher Score: {avg_teacher_score:.2f}/5.0")

        # Cleanup
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def compare_all_models(self, num_tests: int = 5) -> Dict:
        """比较所有训练的模型。"""
        print("🏆 Comparing all trained models...")
        print("="*60)

        comparison_results = {
            "timestamp": time.time(),
            "models": {},
            "comparison": {}
        }

        # Test each model
        for stage, model_name in self.trained_models.items():
            print(f"\n🔄 Testing {stage} model...")
            try:
                results = self.validate_single_model(model_name, num_tests)
                comparison_results["models"][stage] = results
            except Exception as e:
                print(f"❌ Failed to test {stage} model: {e}")
                comparison_results["models"][stage] = {"error": str(e)}

        # Generate comparison
        self._generate_model_comparison(comparison_results)

        return comparison_results

    def _evaluate_variation_similarity(self, original: str, variation: str) -> float:
        """评估变体问题的质量。"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer(max_features=100)
            tfidf_matrix = vectorizer.fit_transform([original, variation])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

            # Optimal similarity range: 0.3-0.7
            if 0.3 <= similarity <= 0.7:
                return 1.0
            elif similarity < 0.3:
                return similarity / 0.3
            else:
                return (1.0 - similarity) / 0.3
        except:
            return 0.5

    def _generate_model_comparison(self, results: Dict):
        """生成模型比较报告。"""
        print(f"\n📊 Model Comparison Report:")
        print("="*60)

        # Extract statistics
        model_stats = {}
        for stage, result in results["models"].items():
            if "error" not in result and "statistics" in result:
                stats = result["statistics"]
                model_stats[stage] = stats

        if not model_stats:
            print("❌ No valid model results to compare")
            return

        # Quality comparison
        print("🎯 Quality Scores:")
        quality_scores = {stage: stats["average_quality_score"] 
                         for stage, stats in model_stats.items()}
        sorted_quality = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)
        
        for i, (stage, score) in enumerate(sorted_quality):
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "🏅"
            print(f"   {rank_emoji} {stage}: {score:.3f}")

        # Speed comparison
        print("\n⚡ Generation Speed:")
        speed_scores = {stage: stats["average_generation_time"] 
                       for stage, stats in model_stats.items()}
        sorted_speed = sorted(speed_scores.items(), key=lambda x: x[1])
        
        for i, (stage, time_val) in enumerate(sorted_speed):
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "🏅"
            print(f"   {rank_emoji} {stage}: {time_val:.2f}s")

        # Teacher scores comparison (if available)
        teacher_scores = {stage: stats["average_teacher_score"] 
                         for stage, stats in model_stats.items() 
                         if stats["average_teacher_score"] is not None}
        
        if teacher_scores:
            print("\n👨‍🏫 Teacher Evaluation:")
            sorted_teacher = sorted(teacher_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (stage, score) in enumerate(sorted_teacher):
                rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else "🏅"
                print(f"   {rank_emoji} {stage}: {score:.2f}/5.0")

        # Overall recommendation
        print(f"\n💡 Recommendations:")
        best_quality = sorted_quality[0][0]
        fastest = sorted_speed[0][0]
        
        if teacher_scores:
            best_teacher = sorted_teacher[0][0]
            print(f"   🎯 Best Quality: {best_quality}")
            print(f"   ⚡ Fastest: {fastest}")
            print(f"   👨‍🏫 Best Teacher Score: {best_teacher}")
            
            if best_quality == best_teacher:
                print(f"   🌟 Recommended for production: {best_quality}")
            else:
                print(f"   🌟 Consider: {best_quality} (quality) or {best_teacher} (teacher approval)")
        else:
            print(f"   🎯 Best Overall: {best_quality}")
            print(f"   ⚡ For speed-critical applications: {fastest}")

    def save_validation_results(self, results: Dict, filename: str = None):
        """保存验证结果。"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"validation_results_{timestamp}.json"

        os.makedirs("validation_results", exist_ok=True)
        filepath = os.path.join("validation_results", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Validation results saved to: {filepath}")
        return filepath
