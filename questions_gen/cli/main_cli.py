# -*- coding: utf-8 -*-
"""Command-line interface for Questions-Gen package."""

import argparse
import sys
import os
from typing import Optional

from ..validation.model_validator import ModelValidator
from ..validation.batch_validator import BatchValidator
from ..validation.quality_evaluator import QualityEvaluator
from ..utils.ollama_manager import OllamaManager
from ..utils.hf_utils import HuggingFaceUtils


def create_parser():
    """创建命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Questions-Gen: Mathematical Problem Generation Package",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  questions-gen validate --model final --tests 5
  questions-gen batch --category algebra --tests 3
  questions-gen ollama --push-all
  questions-gen compare --all-models
  questions-gen quality "Find the derivative of f(x) = x^3 + 2x"
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Validation commands
    validate_parser = subparsers.add_parser('validate', help='Validate individual models')
    validate_parser.add_argument('--model', choices=['stage1', 'stage2', 'final', 'all'], 
                                default='final', help='Model to validate')
    validate_parser.add_argument('--tests', type=int, default=5, 
                                help='Number of test questions')
    validate_parser.add_argument('--save', action='store_true', 
                                help='Save validation results')

    # Batch validation
    batch_parser = subparsers.add_parser('batch', help='Batch validation')
    batch_parser.add_argument('--category', choices=['algebra', 'geometry', 'calculus', 
                                                    'number_theory', 'combinatorics', 
                                                    'analysis', 'all'], 
                             default='all', help='Category to test')
    batch_parser.add_argument('--tests', type=int, default=3, 
                             help='Tests per category')
    batch_parser.add_argument('--models', nargs='+', 
                             help='Specific models to test')
    batch_parser.add_argument('--parallel', action='store_true', 
                             help='Run tests in parallel')
    batch_parser.add_argument('--export-csv', action='store_true', 
                             help='Export results to CSV')

    # Model comparison
    compare_parser = subparsers.add_parser('compare', help='Compare models')
    compare_parser.add_argument('--all-models', action='store_true', 
                               help='Compare all available models')
    compare_parser.add_argument('--tests', type=int, default=3, 
                               help='Number of tests per model')

    # Quality evaluation
    quality_parser = subparsers.add_parser('quality', help='Evaluate question quality')
    quality_parser.add_argument('question', help='Question to evaluate')
    quality_parser.add_argument('--detailed', action='store_true', 
                               help='Show detailed metrics')

    # Ollama integration
    ollama_parser = subparsers.add_parser('ollama', help='Ollama integration')
    ollama_parser.add_argument('--push-all', action='store_true', 
                              help='Push all models to Ollama')
    ollama_parser.add_argument('--push', choices=['stage1', 'stage2', 'final'], 
                              help='Push specific model to Ollama')
    ollama_parser.add_argument('--list', action='store_true', 
                              help='List Questions-Gen models in Ollama')
    ollama_parser.add_argument('--test', type=str, 
                              help='Test specific model in Ollama')
    ollama_parser.add_argument('--force', action='store_true', 
                              help='Force overwrite existing models')
    ollama_parser.add_argument('--guide', action='store_true', 
                              help='Generate usage guide')

    # HuggingFace utilities
    hf_parser = subparsers.add_parser('hf', help='HuggingFace utilities')
    hf_parser.add_argument('--verify', action='store_true', 
                          help='Verify all models exist on HF')
    hf_parser.add_argument('--details', choices=['stage1', 'stage2', 'final'], 
                          help='Get detailed info for specific model')
    hf_parser.add_argument('--compare', action='store_true', 
                          help='Compare all models on HF')
    hf_parser.add_argument('--health', action='store_true', 
                          help='Check model health status')
    hf_parser.add_argument('--model-cards', action='store_true', 
                          help='Generate model cards')
    hf_parser.add_argument('--guide', action='store_true', 
                          help='Generate integration guide')

    # General options
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Verbose output')
    parser.add_argument('--output', '-o', type=str, 
                       help='Output directory for results')

    return parser


def handle_validate_command(args):
    """处理模型验证命令。"""
    print("🧪 Starting model validation...")
    
    validator = ModelValidator()
    
    if args.model == 'all':
        results = validator.compare_all_models(args.tests)
    else:
        model_map = {
            'stage1': validator.trained_models['stage1'],
            'stage2': validator.trained_models['stage2'], 
            'final': validator.trained_models['final']
        }
        model_name = model_map[args.model]
        results = validator.validate_single_model(model_name, args.tests)
    
    if args.save:
        validator.save_validation_results(results)
    
    return results


def handle_batch_command(args):
    """处理批量验证命令。"""
    print("🚀 Starting batch validation...")
    
    batch_validator = BatchValidator()
    
    if args.models:
        # Test specific models
        comparison_results = batch_validator.comparative_batch_validation(
            models=args.models,
            category=args.category,
            tests_per_category=args.tests
        )
    else:
        # Test default models
        comparison_results = batch_validator.comparative_batch_validation(
            category=args.category,
            tests_per_category=args.tests
        )
    
    # Save results
    batch_validator.save_batch_results(comparison_results)
    
    if args.export_csv:
        batch_validator.export_results_to_csv(comparison_results)
    
    return comparison_results


def handle_compare_command(args):
    """处理模型比较命令。"""
    print("📊 Starting model comparison...")
    
    validator = ModelValidator()
    results = validator.compare_all_models(args.tests)
    
    return results


def handle_quality_command(args):
    """处理质量评估命令。"""
    print(f"🔍 Evaluating question quality...")
    print(f"Question: {args.question}")
    
    evaluator = QualityEvaluator()
    evaluation = evaluator.comprehensive_evaluation(args.question)
    
    print(f"\n📊 Quality Evaluation Results:")
    print(f"Overall Score: {evaluation['overall_score']:.3f}")
    print(f"Grade: {evaluation['grade']}")
    
    if args.detailed:
        print(f"\n📋 Detailed Metrics:")
        for metric, data in evaluation['metrics'].items():
            if isinstance(data, dict) and 'score' in data:
                print(f"  {metric}: {data['score']:.3f}")
    
    if evaluation['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in evaluation['recommendations']:
            print(f"  - {rec}")
    
    return evaluation


def handle_ollama_command(args):
    """处理Ollama集成命令。"""
    ollama = OllamaManager()
    
    if not ollama.ollama_available:
        print("❌ Ollama is not installed or not available")
        print("Please install Ollama from: https://ollama.ai")
        return None
    
    if args.push_all:
        print("🚀 Pushing all models to Ollama...")
        results = ollama.push_all_models(force=args.force)
        return results
    
    elif args.push:
        print(f"📦 Pushing {args.push} model to Ollama...")
        result = ollama.push_model_to_ollama(args.push, force=args.force)
        return result
    
    elif args.list:
        print("📋 Listing Questions-Gen models in Ollama...")
        models = ollama.list_questions_gen_models()
        return models
    
    elif args.test:
        print(f"🧪 Testing model {args.test} in Ollama...")
        results = ollama.test_model_in_ollama(args.test)
        return results
    
    elif args.guide:
        print("📚 Generating Ollama usage guide...")
        guide_path = ollama.save_usage_guide()
        return guide_path
    
    else:
        print("Please specify an Ollama action (--push-all, --push, --list, --test, --guide)")
        return None


def handle_hf_command(args):
    """处理HuggingFace工具命令。"""
    hf_utils = HuggingFaceUtils()
    
    if args.verify:
        print("🔍 Verifying models on HuggingFace...")
        results = hf_utils.verify_models_exist()
        return results
    
    elif args.details:
        print(f"📋 Getting details for {args.details} model...")
        details = hf_utils.get_model_details(args.details)
        if details:
            print(f"\n📊 Model Details:")
            print(f"Name: {details['model_name']}")
            print(f"Downloads: {details['downloads']:,}")
            print(f"Likes: {details['likes']}")
            print(f"Size: {details['model_size_gb']} GB")
            print(f"Last Modified: {details['last_modified']}")
        return details
    
    elif args.compare:
        print("📊 Comparing all models on HuggingFace...")
        comparison = hf_utils.compare_all_models()
        return comparison
    
    elif args.health:
        print("🏥 Checking model health status...")
        health = hf_utils.check_model_health()
        return health
    
    elif args.model_cards:
        print("📝 Generating model cards...")
        cards = hf_utils.create_model_cards()
        return cards
    
    elif args.guide:
        print("📚 Generating integration guide...")
        guide_path = hf_utils.save_integration_guide()
        return guide_path
    
    else:
        print("Please specify a HuggingFace action")
        return None


def main():
    """主命令行入口函数。"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Set output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        os.chdir(args.output)
    
    try:
        if args.command == 'validate':
            result = handle_validate_command(args)
        elif args.command == 'batch':
            result = handle_batch_command(args)
        elif args.command == 'compare':
            result = handle_compare_command(args)
        elif args.command == 'quality':
            result = handle_quality_command(args)
        elif args.command == 'ollama':
            result = handle_ollama_command(args)
        elif args.command == 'hf':
            result = handle_hf_command(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
        
        if args.verbose and result:
            print(f"\n🔍 Detailed Results: {result}")
        
        print("\n✅ Command completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Command interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error executing command: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
