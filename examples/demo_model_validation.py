#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Questions-Gen 模型验证演示

基于已训练完成的模型系列进行验证和测试：
- Stage 1: https://huggingface.co/xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged
- Stage 2: https://huggingface.co/xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit
- Final: https://huggingface.co/xingqiang/questions-gen-qwen3-14b-final-merged-16bit
"""

import sys
import os
import time

# 添加当前目录到Python路径
sys.path.insert(0, '.')

# 检查基本导入
print("🔧 检查导入依赖...")
try:
    from questions_gen.models.reward_calculator import RewardCalculator
    from questions_gen.models.deepseek_teacher import DeepSeekTeacher
    print("✅ 基本模块导入成功")
except Exception as e:
    print(f"❌ 基本模块导入失败: {e}")

# 检查高级功能导入
try:
    from questions_gen.validation.model_validator import ModelValidator
    from questions_gen.validation.batch_validator import BatchValidator
    from questions_gen.validation.quality_evaluator import QualityEvaluator
    from questions_gen.utils.hf_utils import HuggingFaceUtils
    FULL_FEATURES = True
    print("✅ 完整功能可用")
except Exception as e:
    print(f"⚠️ 部分功能不可用 (需要安装完整依赖): {e}")
    FULL_FEATURES = False


def demo_basic_quality_evaluation():
    """演示基础质量评估功能"""
    print("\n🔍 演示: 基础质量评估")
    print("-" * 40)
    
    calculator = RewardCalculator()
    
    # 测试问题
    test_questions = [
        "Find all real solutions to the equation x⁴ - 5x² + 6 = 0.",
        "Prove that the square root of 2 is irrational.", 
        "Calculate the derivative of f(x) = x³ + 2x² - 5x + 1.",
        "Solve the integral ∫(2x + 3)dx.",
        "x + 2 = 5"
    ]
    
    print("评估数学问题质量...")
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 问题 {i}: {question}")
        
        # 计算各项分数
        difficulty = calculator.calculate_difficulty(question)
        rigor = calculator.calculate_rigor(question)
        
        # 计算总分 (简化版)
        overall = calculator.calculate_reward(question, [], [])
        
        print(f"   难度: {difficulty:.3f}")
        print(f"   严谨性: {rigor:.3f}")
        print(f"   总分: {overall:.3f}")
        
        # 简单评级
        if overall > 0.7:
            grade = "A (优秀)"
        elif overall > 0.5:
            grade = "B (良好)"
        elif overall > 0.3:
            grade = "C (一般)"
        else:
            grade = "D (需改进)"
        print(f"   等级: {grade}")
    
    return test_questions


def demo_model_info():
    """演示模型信息获取"""
    print("\n🤗 演示: 模型信息验证")
    print("-" * 40)
    
    # 模型列表
    models = {
        "Stage 1": "xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged",
        "Stage 2": "xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit",
        "Final": "xingqiang/questions-gen-qwen3-14b-final-merged-16bit"
    }
    
    if FULL_FEATURES:
        hf_utils = HuggingFaceUtils()
        print("验证HuggingFace模型...")
        
        verification = hf_utils.verify_models_exist()
        
        for stage, model_name in models.items():
            verified = verification.get(stage.lower().replace(' ', ''), False)
            status = "✅" if verified else "❌"
            print(f"{status} {stage}: {model_name}")
            
            if verified:
                try:
                    details = hf_utils.get_model_details(stage.lower().replace(' ', ''))
                    if details:
                        print(f"    📊 下载量: {details['downloads']:,}")
                        print(f"    👍 点赞数: {details['likes']}")
                        if details['model_size_gb']:
                            print(f"    💾 大小: {details['model_size_gb']} GB")
                except Exception as e:
                    print(f"    ⚠️ 详细信息获取失败: {e}")
    else:
        print("📋 模型信息 (无法验证，需要完整依赖):")
        for stage, model_name in models.items():
            print(f"📦 {stage}: {model_name}")
    
    return models


def demo_validation_workflow():
    """演示验证工作流程"""
    print("\n🧪 演示: 验证工作流程")
    print("-" * 40)
    
    print("这是一个完整的模型验证工作流程演示：")
    print("1. 质量评估系统测试")
    print("2. 模型信息验证")
    print("3. 批量验证准备")
    
    # 质量评估演示
    questions = demo_basic_quality_evaluation()
    
    # 模型信息演示  
    models = demo_model_info()
    
    # 工作流程总结
    print(f"\n📊 工作流程总结:")
    print(f"✅ 评估了 {len(questions)} 个测试问题")
    print(f"✅ 检查了 {len(models)} 个训练模型")
    
    if FULL_FEATURES:
        print("✅ 完整验证功能可用")
        print("💡 可以运行: questions-gen validate --model final")
        print("💡 可以运行: questions-gen batch --category algebra")
        print("💡 可以运行: questions-gen ollama --push-all")
    else:
        print("⚠️ 部分功能需要完整依赖")
        print("💡 安装依赖: pip install transformers unsloth datasets")
    
    return {"questions": len(questions), "models": len(models)}


def demo_teacher_model():
    """演示教师模型功能"""
    print("\n👨‍🏫 演示: DeepSeek教师模型")
    print("-" * 40)
    
    teacher = DeepSeekTeacher()
    
    if teacher.client:
        print("✅ DeepSeek-R1 API连接成功")
        
        # 测试评估功能
        sample_problem = "Find the derivative of f(x) = x³ + 2x² - 5x + 1"
        print(f"\n📝 测试问题: {sample_problem}")
        print("🔄 获取教师评估...")
        
        try:
            evaluation = teacher.evaluate_problem(sample_problem)
            print(f"📊 教师评分: {evaluation['overall_score']:.2f}/5.0")
            print(f"📈 难度: {evaluation['difficulty_score']:.1f}")
            print(f"📈 严谨性: {evaluation['rigor_score']:.1f}")
            print(f"📈 创新性: {evaluation['innovation_score']:.1f}")
        except Exception as e:
            print(f"⚠️ 教师评估失败: {e}")
    else:
        print("❌ DeepSeek-R1 API不可用")
        print("💡 需要设置 DEEPSEEK_API_KEY 环境变量")
        print("💡 或者API服务暂时不可用")


def main():
    """主演示函数"""
    print("🎯 Questions-Gen 模型验证演示")
    print("="*60)
    print("基于已训练完成的模型系列进行验证测试")
    print()
    
    try:
        # 基础功能演示
        print("🔧 运行基础功能测试...")
        results = demo_validation_workflow()
        
        # 教师模型演示
        demo_teacher_model()
        
        # 总结
        print(f"\n🎉 演示完成!")
        print(f"📊 测试结果:")
        print(f"   - 问题评估: {results['questions']} 个")
        print(f"   - 模型检查: {results['models']} 个")
        print(f"   - 功能状态: {'完整' if FULL_FEATURES else '基础'}")
        
        print(f"\n💡 下一步建议:")
        if FULL_FEATURES:
            print("1. 运行完整验证: python examples/demo_validation.py")
            print("2. 推送到Ollama: python demo_ollama_push.py")
            print("3. 使用CLI工具: questions-gen --help")
        else:
            print("1. 安装完整依赖获得全部功能")
            print("2. 设置HF_TOKEN环境变量")
            print("3. 设置DEEPSEEK_API_KEY环境变量")
        
    except KeyboardInterrupt:
        print(f"\n❌ 演示被用户中断")
    except Exception as e:
        print(f"❌ 演示过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
