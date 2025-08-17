# -*- coding: utf-8 -*-
"""Batch validation for large-scale model testing."""

import os
import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from .model_validator import ModelValidator


class BatchValidator:
    """Large-scale batch validation for Questions-Gen models."""

    def __init__(self, max_workers: int = 2):
        self.validator = ModelValidator()
        self.max_workers = max_workers
        
        # 扩展测试类别
        self.test_categories = {
            "algebra": [
                "Generate a challenging algebra competition problem:",
                "Create an algebraic equation with complex solutions:",
                "Design a polynomial problem for advanced students:",
                "Formulate an abstract algebra question:",
                "Generate a system of equations problem:"
            ],
            "geometry": [
                "Create a geometry problem suitable for math olympiad:",
                "Design a geometric proof problem:",
                "Generate a coordinate geometry challenge:",
                "Create a solid geometry problem:",
                "Formulate a trigonometry-based geometry question:"
            ],
            "calculus": [
                "Design a calculus problem with moderate difficulty:",
                "Create a limit evaluation challenge:",
                "Generate an integration problem:",
                "Design a differential equations question:",
                "Create an optimization problem:"
            ],
            "number_theory": [
                "Formulate a number theory competition question:",
                "Create a prime number theory problem:",
                "Generate a modular arithmetic challenge:",
                "Design a Diophantine equation problem:",
                "Create a divisibility theory question:"
            ],
            "combinatorics": [
                "Develop a combinatorics problem for advanced students:",
                "Create a permutation and combination challenge:",
                "Generate a graph theory problem:",
                "Design a counting principle question:",
                "Create a probability combinatorics problem:"
            ],
            "analysis": [
                "Create a proof-based analysis problem:",
                "Generate a real analysis challenge:",
                "Design a sequence and series problem:",
                "Create a function analysis question:",
                "Formulate a convergence problem:"
            ]
        }

    def batch_validate_model(self, model_name: str, 
                           category: str = "all", 
                           tests_per_category: int = 3,
                           parallel: bool = False) -> Dict:
        """批量验证单个模型。"""
        print(f"🚀 Starting batch validation for: {model_name}")
        print(f"📋 Category: {category}, Tests per category: {tests_per_category}")
        print("="*60)

        # Determine which categories to test
        if category == "all":
            categories_to_test = self.test_categories.keys()
        elif category in self.test_categories:
            categories_to_test = [category]
        else:
            raise ValueError(f"Unknown category: {category}")

        results = {
            "model_name": model_name,
            "timestamp": time.time(),
            "test_configuration": {
                "category": category,
                "tests_per_category": tests_per_category,
                "parallel": parallel
            },
            "category_results": {},
            "overall_statistics": {}
        }

        # Load model once for all tests
        print(f"🔄 Loading model: {model_name}")
        model, tokenizer = self.validator.load_model_from_hf(model_name)
        if model is None:
            return {"error": f"Failed to load model {model_name}"}

        total_tests = 0
        total_quality = 0
        total_time = 0
        category_scores = {}

        # Test each category
        for cat_name in categories_to_test:
            print(f"\n📂 Testing category: {cat_name.upper()}")
            print("-" * 40)

            cat_results = self._test_category(
                model, tokenizer, cat_name, tests_per_category, parallel
            )

            results["category_results"][cat_name] = cat_results
            category_scores[cat_name] = cat_results["statistics"]["average_quality"]

            # Update totals
            total_tests += cat_results["statistics"]["total_tests"]
            total_quality += cat_results["statistics"]["total_quality"]
            total_time += cat_results["statistics"]["total_time"]

        # Calculate overall statistics
        results["overall_statistics"] = {
            "total_tests": total_tests,
            "average_quality": total_quality / total_tests if total_tests > 0 else 0,
            "total_time": total_time,
            "average_time_per_test": total_time / total_tests if total_tests > 0 else 0,
            "category_scores": category_scores,
            "best_category": max(category_scores, key=category_scores.get) if category_scores else None,
            "worst_category": min(category_scores, key=category_scores.get) if category_scores else None
        }

        # Print summary
        self._print_batch_summary(results)

        # Cleanup
        del model, tokenizer
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        return results

    def _test_category(self, model, tokenizer, category: str, 
                      num_tests: int, parallel: bool = False) -> Dict:
        """测试特定类别的问题。"""
        prompts = self.test_categories[category][:num_tests]
        
        cat_results = {
            "category": category,
            "test_results": [],
            "statistics": {}
        }

        if parallel and num_tests > 1:
            # Parallel execution
            results_list = self._run_parallel_tests(model, tokenizer, prompts)
        else:
            # Sequential execution
            results_list = self._run_sequential_tests(model, tokenizer, prompts)

        cat_results["test_results"] = results_list

        # Calculate category statistics
        qualities = [r["quality_score"] for r in results_list if r["quality_score"] > 0]
        times = [r["generation_time"] for r in results_list if r["generation_time"] > 0]
        
        cat_results["statistics"] = {
            "total_tests": len(results_list),
            "successful_tests": len(qualities),
            "average_quality": np.mean(qualities) if qualities else 0,
            "std_quality": np.std(qualities) if qualities else 0,
            "min_quality": np.min(qualities) if qualities else 0,
            "max_quality": np.max(qualities) if qualities else 0,
            "total_quality": sum(qualities),
            "total_time": sum(times),
            "average_time": np.mean(times) if times else 0,
            "std_time": np.std(times) if times else 0
        }

        return cat_results

    def _run_sequential_tests(self, model, tokenizer, prompts: List[str]) -> List[Dict]:
        """顺序执行测试。"""
        results = []
        
        for i, prompt in enumerate(prompts):
            print(f"   Test {i+1}/{len(prompts)}: {prompt[:30]}...")
            
            start_time = time.time()
            question = self.validator.generate_question(model, tokenizer, prompt)
            generation_time = time.time() - start_time
            
            quality_score = self.validator.reward_calculator.calculate_reward(question, [], [])
            
            result = {
                "test_id": i + 1,
                "prompt": prompt,
                "generated_question": question,
                "quality_score": quality_score,
                "generation_time": generation_time,
                "success": len(question) > 20
            }
            results.append(result)
            
            print(f"      Quality: {quality_score:.3f}, Time: {generation_time:.2f}s")

        return results

    def _run_parallel_tests(self, model, tokenizer, prompts: List[str]) -> List[Dict]:
        """并行执行测试（注意：需要小心GPU内存）。"""
        print("   ⚡ Running parallel tests...")
        results = []
        
        # Note: Parallel generation with single GPU can be tricky
        # For now, implement a simple batch approach
        for i, prompt in enumerate(prompts):
            start_time = time.time()
            question = self.validator.generate_question(model, tokenizer, prompt)
            generation_time = time.time() - start_time
            
            quality_score = self.validator.reward_calculator.calculate_reward(question, [], [])
            
            result = {
                "test_id": i + 1,
                "prompt": prompt,
                "generated_question": question,
                "quality_score": quality_score,
                "generation_time": generation_time,
                "success": len(question) > 20
            }
            results.append(result)

        return results

    def comparative_batch_validation(self, 
                                   models: List[str] = None,
                                   category: str = "all",
                                   tests_per_category: int = 2) -> Dict:
        """比较多个模型的批量验证。"""
        if models is None:
            models = list(self.validator.trained_models.values())

        print(f"🏆 Comparative Batch Validation")
        print(f"📋 Models: {len(models)}, Category: {category}")
        print("="*60)

        comparison_results = {
            "timestamp": time.time(),
            "configuration": {
                "models": models,
                "category": category,
                "tests_per_category": tests_per_category
            },
            "model_results": {},
            "comparison_analysis": {}
        }

        # Test each model
        for i, model_name in enumerate(models):
            print(f"\n🔄 [{i+1}/{len(models)}] Testing: {model_name}")
            try:
                results = self.batch_validate_model(
                    model_name, category, tests_per_category, parallel=False
                )
                comparison_results["model_results"][model_name] = results
            except Exception as e:
                print(f"❌ Failed to test {model_name}: {e}")
                comparison_results["model_results"][model_name] = {"error": str(e)}

        # Generate comparison analysis
        self._generate_comparative_analysis(comparison_results)

        return comparison_results

    def _generate_comparative_analysis(self, results: Dict):
        """生成比较分析报告。"""
        print(f"\n📊 Comparative Analysis Report")
        print("="*60)

        # Extract valid results
        valid_results = {name: result for name, result in results["model_results"].items() 
                        if "error" not in result}

        if not valid_results:
            print("❌ No valid results to analyze")
            return

        # Overall performance comparison
        print("🎯 Overall Performance:")
        overall_scores = {}
        for name, result in valid_results.items():
            if "overall_statistics" in result:
                score = result["overall_statistics"]["average_quality"]
                overall_scores[name] = score
                model_short = name.split('/')[-1] if '/' in name else name
                print(f"   {model_short}: {score:.3f}")

        # Category-wise comparison
        if valid_results:
            first_result = next(iter(valid_results.values()))
            if "category_results" in first_result:
                categories = first_result["category_results"].keys()
                
                print(f"\n📂 Category Performance:")
                for category in categories:
                    print(f"\n   {category.upper()}:")
                    cat_scores = {}
                    for name, result in valid_results.items():
                        if category in result["category_results"]:
                            score = result["category_results"][category]["statistics"]["average_quality"]
                            cat_scores[name] = score
                            model_short = name.split('/')[-1] if '/' in name else name
                            print(f"      {model_short}: {score:.3f}")

        # Speed comparison
        print(f"\n⚡ Speed Performance:")
        for name, result in valid_results.items():
            if "overall_statistics" in result:
                avg_time = result["overall_statistics"]["average_time_per_test"]
                model_short = name.split('/')[-1] if '/' in name else name
                print(f"   {model_short}: {avg_time:.2f}s/test")

        # Recommendations
        print(f"\n💡 Recommendations:")
        if overall_scores:
            best_model = max(overall_scores, key=overall_scores.get)
            best_short = best_model.split('/')[-1] if '/' in best_model else best_model
            print(f"   🌟 Highest Quality: {best_short} ({overall_scores[best_model]:.3f})")

    def _print_batch_summary(self, results: Dict):
        """打印批量验证摘要。"""
        print(f"\n📊 Batch Validation Summary:")
        print("="*40)
        
        stats = results["overall_statistics"]
        print(f"Total Tests: {stats['total_tests']}")
        print(f"Average Quality: {stats['average_quality']:.3f}")
        print(f"Total Time: {stats['total_time']:.2f}s")
        print(f"Avg Time/Test: {stats['average_time_per_test']:.2f}s")
        
        if stats['best_category']:
            print(f"Best Category: {stats['best_category']} ({stats['category_scores'][stats['best_category']]:.3f})")
        if stats['worst_category']:
            print(f"Worst Category: {stats['worst_category']} ({stats['category_scores'][stats['worst_category']]:.3f})")

    def export_results_to_csv(self, results: Dict, filename: str = None) -> str:
        """导出结果到CSV文件。"""
        if filename is None:
            timestamp = int(time.time())
            filename = f"batch_validation_{timestamp}.csv"

        os.makedirs("validation_results", exist_ok=True)
        filepath = os.path.join("validation_results", filename)

        # Prepare data for CSV
        csv_data = []
        
        if "category_results" in results:
            for category, cat_result in results["category_results"].items():
                for test in cat_result["test_results"]:
                    row = {
                        "model_name": results.get("model_name", "unknown"),
                        "category": category,
                        "test_id": test["test_id"],
                        "prompt": test["prompt"][:100],  # Truncate for CSV
                        "generated_question": test["generated_question"][:200],  # Truncate
                        "quality_score": test["quality_score"],
                        "generation_time": test["generation_time"],
                        "success": test["success"]
                    }
                    csv_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(filepath, index=False)
        
        print(f"📄 Results exported to CSV: {filepath}")
        return filepath

    def save_batch_results(self, results: Dict, filename: str = None):
        """保存批量验证结果。"""
        if filename is None:
            timestamp = int(time.time())
            if "model_name" in results:
                model_short = results["model_name"].split('/')[-1]
                filename = f"batch_validation_{model_short}_{timestamp}.json"
            else:
                filename = f"batch_validation_{timestamp}.json"

        os.makedirs("validation_results", exist_ok=True)
        filepath = os.path.join("validation_results", filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Batch results saved to: {filepath}")
        
        # Also save CSV version
        self.export_results_to_csv(results, filename.replace('.json', '.csv'))
        
        return filepath
