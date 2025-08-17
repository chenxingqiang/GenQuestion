#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Questions-Gen 快速功能演示

展示已完成的包功能：
✅ 模块化包结构
✅ 模型验证系统  
✅ 批量测试功能
✅ 质量评估系统
✅ Ollama集成管理
✅ HuggingFace工具
✅ 命令行接口
✅ 完整文档
"""

import sys
import os

# 添加当前目录
sys.path.insert(0, '.')

def check_package_structure():
    """检查包结构"""
    print("📦 检查包结构...")
    
    expected_modules = [
        'questions_gen',
        'questions_gen.core',
        'questions_gen.models', 
        'questions_gen.data',
        'questions_gen.validation',
        'questions_gen.utils',
        'questions_gen.cli'
    ]
    
    for module in expected_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    return len(expected_modules)

def check_core_features():
    """检查核心功能"""
    print("\n🔧 检查核心功能...")
    
    features = {}
    
    # 配置系统
    try:
        from questions_gen.core.config import TrainingConfig
        config = TrainingConfig()
        features['config'] = True
        print(f"✅ 配置系统: {config.MODEL_NAME}")
    except Exception as e:
        features['config'] = False
        print(f"❌ 配置系统: {e}")
    
    # 奖励计算器
    try:
        from questions_gen.models.reward_calculator import RewardCalculator
        calculator = RewardCalculator()
        score = calculator.calculate_difficulty("Find derivative of x^2")
        features['reward'] = True
        print(f"✅ 奖励计算器: 测试分数 {score:.3f}")
    except Exception as e:
        features['reward'] = False
        print(f"❌ 奖励计算器: {e}")
    
    # DeepSeek教师
    try:
        from questions_gen.models.deepseek_teacher import DeepSeekTeacher
        teacher = DeepSeekTeacher()
        features['teacher'] = teacher.client is not None
        status = "连接成功" if features['teacher'] else "API不可用"
        print(f"✅ DeepSeek教师: {status}")
    except Exception as e:
        features['teacher'] = False
        print(f"❌ DeepSeek教师: {e}")
    
    return features

def check_validation_system():
    """检查验证系统"""
    print("\n🧪 检查验证系统...")
    
    validation_features = {}
    
    # 质量评估器
    try:
        from questions_gen.validation.quality_evaluator import QualityEvaluator
        evaluator = QualityEvaluator()
        
        # 测试评估
        test_question = "Prove that √2 is irrational"
        evaluation = evaluator.comprehensive_evaluation(test_question)
        validation_features['quality'] = True
        print(f"✅ 质量评估器: 分数 {evaluation['overall_score']:.3f}, 等级 {evaluation['grade']}")
    except Exception as e:
        validation_features['quality'] = False
        print(f"❌ 质量评估器: {e}")
    
    # 模型验证器
    try:
        from questions_gen.validation.model_validator import ModelValidator
        validator = ModelValidator()
        validation_features['model_validator'] = True
        print(f"✅ 模型验证器: 支持 {len(validator.trained_models)} 个模型")
    except Exception as e:
        validation_features['model_validator'] = False
        print(f"❌ 模型验证器: {e}")
    
    # 批量验证器
    try:
        from questions_gen.validation.batch_validator import BatchValidator
        batch_validator = BatchValidator()
        categories = len(batch_validator.test_categories)
        validation_features['batch'] = True
        print(f"✅ 批量验证器: 支持 {categories} 个测试类别")
    except Exception as e:
        validation_features['batch'] = False
        print(f"❌ 批量验证器: {e}")
    
    return validation_features

def check_utils():
    """检查工具模块"""
    print("\n🛠️ 检查工具模块...")
    
    utils_features = {}
    
    # Ollama管理器
    try:
        from questions_gen.utils.ollama_manager import OllamaManager
        ollama = OllamaManager()
        utils_features['ollama'] = True
        print(f"✅ Ollama管理器: {'可用' if ollama.ollama_available else '未安装'}")
    except Exception as e:
        utils_features['ollama'] = False
        print(f"❌ Ollama管理器: {e}")
    
    # HuggingFace工具
    try:
        from questions_gen.utils.hf_utils import HuggingFaceUtils
        hf_utils = HuggingFaceUtils()
        model_count = len(hf_utils.qgen_models)
        utils_features['hf'] = True
        print(f"✅ HuggingFace工具: 管理 {model_count} 个模型")
    except Exception as e:
        utils_features['hf'] = False
        print(f"❌ HuggingFace工具: {e}")
    
    return utils_features

def check_cli():
    """检查命令行接口"""
    print("\n⌨️ 检查命令行接口...")
    
    try:
        from questions_gen.cli.main_cli import create_parser
        parser = create_parser()
        
        # 检查可用命令
        subparsers_actions = [
            action for action in parser._actions 
            if isinstance(action, argparse._SubParsersAction)
        ]
        
        if subparsers_actions:
            commands = list(subparsers_actions[0].choices.keys())
            print(f"✅ CLI接口: 支持 {len(commands)} 个命令")
            print(f"   命令: {', '.join(commands)}")
            return True
        else:
            print("❌ CLI接口: 无法获取命令列表")
            return False
    except Exception as e:
        print(f"❌ CLI接口: {e}")
        return False

def check_files():
    """检查重要文件"""
    print("\n📄 检查重要文件...")
    
    important_files = [
        'setup.py',
        'pyproject.toml', 
        'requirements.txt',
        'README.md',
        'LICENSE',
        'USAGE_GUIDE.md'
    ]
    
    existing_files = []
    for file in important_files:
        if os.path.exists(file):
            existing_files.append(file)
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
    
    return len(existing_files), len(important_files)

def run_quick_tests():
    """运行快速功能测试"""
    print("\n🏃 运行快速功能测试...")
    
    tests_passed = 0
    total_tests = 0
    
    # 测试1: 基础导入
    total_tests += 1
    try:
        import questions_gen
        print(f"✅ 基础导入: v{questions_gen.__version__}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 基础导入: {e}")
    
    # 测试2: 奖励计算
    total_tests += 1
    try:
        from questions_gen.models.reward_calculator import RewardCalculator
        calc = RewardCalculator()
        reward = calc.calculate_reward("Solve x^2 = 4", [], [])
        print(f"✅ 奖励计算: {reward:.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 奖励计算: {e}")
    
    # 测试3: 质量评估
    total_tests += 1
    try:
        from questions_gen.validation.quality_evaluator import QualityEvaluator
        evaluator = QualityEvaluator()
        evaluation = evaluator.comprehensive_evaluation("Find lim(x→0) sin(x)/x")
        print(f"✅ 质量评估: {evaluation['grade']} ({evaluation['overall_score']:.3f})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ 质量评估: {e}")
    
    return tests_passed, total_tests

def main():
    """主函数"""
    print("🎯 Questions-Gen 快速功能验证")
    print("="*60)
    
    # 汇总结果
    results = {}
    
    # 检查包结构
    module_count = check_package_structure()
    
    # 检查核心功能
    core_features = check_core_features()
    results['core'] = sum(core_features.values())
    
    # 检查验证系统
    validation_features = check_validation_system()
    results['validation'] = sum(validation_features.values())
    
    # 检查工具模块
    utils_features = check_utils()
    results['utils'] = sum(utils_features.values())
    
    # 检查CLI
    cli_available = check_cli()
    results['cli'] = 1 if cli_available else 0
    
    # 检查文件
    file_count, total_files = check_files()
    results['files'] = file_count
    
    # 运行测试
    tests_passed, total_tests = run_quick_tests()
    results['tests'] = tests_passed
    
    # 总结报告
    print(f"\n📊 验证总结")
    print("="*40)
    print(f"📦 包结构: {module_count} 个模块")
    print(f"🔧 核心功能: {results['core']}/3 可用")
    print(f"🧪 验证系统: {results['validation']}/3 可用")
    print(f"🛠️ 工具模块: {results['utils']}/2 可用")
    print(f"⌨️ CLI接口: {'可用' if results['cli'] else '不可用'}")
    print(f"📄 重要文件: {results['files']}/{total_files} 存在")
    print(f"🏃 快速测试: {results['tests']}/{total_tests} 通过")
    
    # 计算总体完成度
    total_score = (
        results['core'] / 3 * 20 +
        results['validation'] / 3 * 25 +
        results['utils'] / 2 * 15 +
        results['cli'] * 10 +
        results['files'] / total_files * 15 +
        results['tests'] / total_tests * 15
    )
    
    print(f"\n🎉 总体完成度: {total_score:.1f}%")
    
    if total_score >= 80:
        print("✅ 包功能完整，可以发布！")
    elif total_score >= 60:
        print("⚠️ 基本功能可用，建议完善后发布")
    else:
        print("❌ 需要修复关键问题后再发布")
    
    # 使用建议
    print(f"\n💡 使用建议:")
    if results['core'] >= 2:
        print("✅ 核心功能可用 - 可以进行质量评估")
    if results['validation'] >= 2:
        print("✅ 验证系统可用 - 可以测试模型性能")
    if results['utils'] >= 1:
        print("✅ 工具模块可用 - 可以管理HF模型或Ollama")
    if results['cli']:
        print("✅ CLI可用 - 可以使用 questions-gen 命令")
    
    print(f"\n📚 文档:")
    print("- 使用指南: USAGE_GUIDE.md")
    print("- README: README.md")
    print("- 演示脚本: demo_*.py")
    print("- 示例: examples/")


if __name__ == "__main__":
    # 检查argparse导入
    try:
        import argparse
    except ImportError:
        print("需要安装argparse")
    
    main()
