#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Questions-Gen Package Demo: Model Validation and Testing

This script demonstrates the comprehensive capabilities of the Questions-Gen package
including model validation, batch testing, quality evaluation, and Ollama integration.
"""

import os
import sys
import argparse

# Add the parent directory to Python path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from questions_gen.validation import ModelValidator, BatchValidator, QualityEvaluator
from questions_gen.utils import OllamaManager, HuggingFaceUtils


def demo_model_validation():
    """演示模型验证功能。"""
    print("🧪 Demo: Model Validation")
    print("="*50)

    validator = ModelValidator()

    # Validate the final model with 3 test questions
    print("Testing the final model...")
    results = validator.validate_single_model(
        "xingqiang/questions-gen-qwen3-14b-final-merged-16bit",
        num_tests=3
    )

    print(f"\n📊 Validation Results:")
    stats = results["statistics"]
    print(f"Average Quality Score: {stats['average_quality_score']:.3f}")
    print(f"Average Generation Time: {stats['average_generation_time']:.2f}s")
    if stats['average_teacher_score']:
        print(f"Average Teacher Score: {stats['average_teacher_score']:.2f}/5.0")

    return results


def demo_batch_validation():
    """演示批量验证功能。"""
    print("\n🚀 Demo: Batch Validation")
    print("="*50)

    batch_validator = BatchValidator()

    # Run batch validation on algebra category
    print("Running batch validation on algebra problems...")
    results = batch_validator.batch_validate_model(
        "xingqiang/questions-gen-qwen3-14b-final-merged-16bit",
        category="algebra",
        tests_per_category=2
    )

    # Show summary
    stats = results["overall_statistics"]
    print(f"\n📊 Batch Results:")
    print(f"Total Tests: {stats['total_tests']}")
    print(f"Average Quality: {stats['average_quality']:.3f}")
    print(f"Best Category: {stats['best_category']}")

    return results


def demo_quality_evaluation():
    """演示质量评估功能。"""
    print("\n🔍 Demo: Quality Evaluation")
    print("="*50)

    evaluator = QualityEvaluator()

    # Test questions for evaluation
    test_questions = [
        "Find all real solutions to the equation x⁴ - 5x² + 6 = 0.",
        "Prove that the square root of 2 is irrational.",
        "Calculate the derivative of f(x) = x³ + 2x² - 5x + 1."
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Question {i}: {question}")
        evaluation = evaluator.comprehensive_evaluation(question)

        print(f"   Overall Score: {evaluation['overall_score']:.3f}")
        print(f"   Grade: {evaluation['grade']}")

        if evaluation['recommendations']:
            print(f"   Recommendations: {len(evaluation['recommendations'])} items")

    return test_questions


def demo_model_comparison():
    """演示模型比较功能。"""
    print("\n📊 Demo: Model Comparison")
    print("="*50)

    validator = ModelValidator()

    print("Comparing all available models...")
    results = validator.compare_all_models(num_tests=2)

    print("\nComparison completed. Check the detailed output above.")
    return results


def demo_hf_utilities():
    """演示HuggingFace工具功能。"""
    print("\n🤗 Demo: HuggingFace Utilities")
    print("="*50)

    hf_utils = HuggingFaceUtils()

    # Verify models exist
    print("Verifying models on HuggingFace...")
    verification = hf_utils.verify_models_exist()

    for stage, exists in verification.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {stage} model")

    # Get model comparison
    if all(verification.values()):
        print("\nGetting model comparison...")
        comparison = hf_utils.compare_all_models()

    return verification


def demo_ollama_integration():
    """演示Ollama集成功能。"""
    print("\n🦙 Demo: Ollama Integration")
    print("="*50)

    ollama = OllamaManager()

    if not ollama.ollama_available:
        print("❌ Ollama not available. Skipping Ollama demo.")
        print("💡 Install Ollama from: https://ollama.ai")
        return None

    # List existing models
    print("Checking for existing Questions-Gen models in Ollama...")
    existing_models = ollama.list_questions_gen_models()

    if not existing_models:
        print("No Questions-Gen models found in Ollama.")
        print("💡 Use 'questions-gen ollama --push-all' to deploy models")
    else:
        # Test an existing model
        test_model = existing_models[0]
        print(f"\nTesting model: {test_model}")

        test_results = ollama.test_model_in_ollama(
            test_model,
            test_prompts=["Generate a simple algebra problem:"]
        )

        print(f"Test success rate: {test_results['success_rate']:.1%}")

    return existing_models


def demo_comprehensive_workflow():
    """演示完整的工作流程。"""
    print("\n🌟 Demo: Comprehensive Workflow")
    print("="*50)

    print("This workflow demonstrates a typical use case:")
    print("1. Validate models")
    print("2. Run quality assessment")
    print("3. Compare performance")
    print("4. Deploy to Ollama")

    workflow_results = {}

    # Step 1: Quick validation
    print("\n🔄 Step 1: Quick Model Validation")
    validator = ModelValidator()
    validation_results = validator.validate_single_model(
        "xingqiang/questions-gen-qwen3-14b-final-merged-16bit",
        num_tests=2
    )
    workflow_results['validation'] = validation_results

    # Step 2: Quality assessment
    print("\n🔄 Step 2: Quality Assessment")
    evaluator = QualityEvaluator()
    sample_question = "Find the derivative of f(x) = x³ + 2x² - 5x + 1"
    quality_eval = evaluator.comprehensive_evaluation(sample_question)
    workflow_results['quality'] = quality_eval

    print(f"   Sample quality score: {quality_eval['overall_score']:.3f}")

    # Step 3: HF verification
    print("\n🔄 Step 3: HuggingFace Verification")
    hf_utils = HuggingFaceUtils()
    hf_status = hf_utils.verify_models_exist()
    workflow_results['hf_status'] = hf_status

    verified_models = sum(hf_status.values())
    print(f"   Verified models: {verified_models}/3")

    # Step 4: Ollama check
    print("\n🔄 Step 4: Ollama Availability Check")
    ollama = OllamaManager()
    if ollama.ollama_available:
        ollama_models = ollama.list_questions_gen_models()
        workflow_results['ollama'] = ollama_models
        print(f"   Ollama models available: {len(ollama_models)}")
    else:
        print("   Ollama not available")
        workflow_results['ollama'] = []

    print("\n✅ Comprehensive workflow completed!")
    return workflow_results


def main():
    """主演示函数。"""
    parser = argparse.ArgumentParser(description="Questions-Gen Package Demo")
    parser.add_argument('--demo', choices=[
        'validation', 'batch', 'quality', 'comparison', 'hf', 'ollama', 'workflow', 'all'
    ], default='workflow', help='Which demo to run')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    args = parser.parse_args()

    print("🎯 Questions-Gen Package Demo")
    print("="*60)
    print("This demo showcases the capabilities of the Questions-Gen package.")
    print("For full functionality, ensure you have:")
    print("- HuggingFace token (HF_TOKEN environment variable)")
    print("- Ollama installed (optional, for local deployment)")
    print("- Sufficient GPU memory for model loading")
    print()

    demos = {
        'validation': demo_model_validation,
        'batch': demo_batch_validation,
        'quality': demo_quality_evaluation,
        'comparison': demo_model_comparison,
        'hf': demo_hf_utilities,
        'ollama': demo_ollama_integration,
        'workflow': demo_comprehensive_workflow
    }

    try:
        if args.demo == 'all':
            results = {}
            for demo_name, demo_func in demos.items():
                print(f"\n{'='*60}")
                print(f"Running demo: {demo_name}")
                results[demo_name] = demo_func()
        else:
            results = demos[args.demo]()

        print(f"\n🎉 Demo completed successfully!")
        if args.verbose and results:
            print(f"\n🔍 Detailed results available in return value")

    except KeyboardInterrupt:
        print(f"\n❌ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
