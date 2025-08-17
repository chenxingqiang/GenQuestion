#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Questions-Gen Ollama推送演示

这个脚本演示如何将已训练的HuggingFace模型推送到Ollama进行本地部署。
基于你已有的三个模型：
- xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged
- xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit  
- xingqiang/questions-gen-qwen3-14b-final-merged-16bit
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, '.')

# 尝试导入（可能需要安装依赖）
try:
    from questions_gen.utils.ollama_manager import OllamaManager
    from questions_gen.utils.hf_utils import HuggingFaceUtils
    print("✅ Questions-Gen包导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("💡 请先安装依赖: pip install transformers huggingface_hub")
    sys.exit(1)


def main():
    """主演示流程"""
    print("🦙 Questions-Gen Ollama 推送演示")
    print("="*60)
    
    # 1. 检查HuggingFace模型状态
    print("🔍 Step 1: 检查HuggingFace模型状态")
    hf_utils = HuggingFaceUtils()
    
    print("验证已训练模型...")
    verification = hf_utils.verify_models_exist()
    
    all_verified = all(verification.values())
    if all_verified:
        print("✅ 所有模型验证通过！")
    else:
        print("⚠️ 部分模型验证失败")
        for stage, verified in verification.items():
            status = "✅" if verified else "❌"
            print(f"   {status} {stage} 模型")
    
    # 2. 检查Ollama环境
    print(f"\n🔧 Step 2: 检查Ollama环境")
    ollama = OllamaManager()
    
    if not ollama.ollama_available:
        print("❌ Ollama未安装或不可用")
        print("📋 安装说明:")
        print("   macOS: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   Windows: 下载安装包从 https://ollama.ai")
        print("\n💡 安装后请重新运行此脚本")
        return
    
    print("✅ Ollama环境检查通过")
    
    # 3. 检查现有模型
    print(f"\n📋 Step 3: 检查现有Ollama模型")
    existing_models = ollama.list_questions_gen_models()
    
    if existing_models:
        print(f"发现 {len(existing_models)} 个已存在的Questions-Gen模型:")
        for model in existing_models:
            print(f"   ✅ {model}")
    else:
        print("📭 未发现已存在的Questions-Gen模型")
    
    # 4. 推送模型选择
    print(f"\n🚀 Step 4: 模型推送选项")
    print("可用的推送选项:")
    print("1. 推送所有模型 (stage1, stage2, final)")
    print("2. 推送单个模型")
    print("3. 生成使用指南")
    print("4. 测试现有模型")
    
    # 推送演示
    if input("\n是否要推送所有模型到Ollama? (y/N): ").lower().startswith('y'):
        print("\n🔄 推送所有模型到Ollama...")
        results = ollama.push_all_models(force=True)
        
        # 显示结果
        successful = sum(results.values())
        total = len(results)
        print(f"\n📊 推送结果: {successful}/{total} 成功")
        
        for stage, success in results.items():
            status = "✅" if success else "❌"
            print(f"   {status} {stage} 模型")
        
        if successful > 0:
            print(f"\n🎉 成功推送 {successful} 个模型!")
            print("💡 使用方法:")
            print("   ollama run questions-gen-final 'Generate a calculus problem:'")
            print("   ollama run questions-gen-stage2 'Create a geometry problem:'")
    
    # 使用指南生成
    elif input("是否生成Ollama使用指南? (y/N): ").lower().startswith('y'):
        print("\n📚 生成使用指南...")
        guide_path = ollama.save_usage_guide()
        print(f"✅ 使用指南已保存到: {guide_path}")
    
    # 测试现有模型
    elif existing_models and input("是否测试现有模型? (y/N): ").lower().startswith('y'):
        test_model = existing_models[0]
        print(f"\n🧪 测试模型: {test_model}")
        
        test_results = ollama.test_model_in_ollama(
            test_model,
            test_prompts=[
                "Generate a simple algebra problem:",
                "Create a calculus question:"
            ]
        )
        
        print(f"✅ 测试完成! 成功率: {test_results['success_rate']:.1%}")
    
    else:
        print("📋 演示完成，未执行推送操作")
    
    # 5. 总结
    print(f"\n📋 总结:")
    print("✅ HuggingFace模型验证完成")
    print("✅ Ollama环境检查完成")
    print("✅ 推送功能演示完成")
    
    print(f"\n💡 后续使用建议:")
    print("1. 使用命令行工具: questions-gen ollama --push-all")
    print("2. 查看使用指南: questions-gen ollama --guide")
    print("3. 测试模型: questions-gen ollama --test questions-gen-final")
    print("4. API调用: curl http://localhost:11434/api/generate ...")


if __name__ == "__main__":
    main()
