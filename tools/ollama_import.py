#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从HuggingFace导入模型到Ollama的脚本
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """运行命令并返回结果"""
    print(f"🔄 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} 成功!")
            if result.stdout.strip():
                print(f"   输出: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} 失败!")
            if result.stderr.strip():
                print(f"   错误: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        return False

def create_modelfile(hf_model_name, model_name):
    """创建Modelfile"""
    modelfile_content = f"""FROM {hf_model_name}

TEMPLATE \"\"\"<|im_start|>system
You are a helpful assistant that generates high-quality mathematical problems.
<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}
<|im_end|>
<|im_start|>assistant
\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
"""
    
    filename = f"{model_name}.modelfile"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"📝 创建Modelfile: {filename}")
    return filename

def import_model_to_ollama(hf_model_name, local_name):
    """将HuggingFace模型导入到Ollama"""
    print(f"\n🚀 导入模型: {hf_model_name} -> {local_name}")
    print("="*60)
    
    # 方法1: 直接使用ollama pull（如果模型在Ollama Hub中）
    print(f"📥 尝试直接拉取模型...")
    cmd = f"ollama pull {hf_model_name}"
    if run_command(cmd, f"直接拉取 {hf_model_name}"):
        # 重命名模型
        cmd = f"ollama cp {hf_model_name} {local_name}"
        return run_command(cmd, f"重命名为 {local_name}")
    
    # 方法2: 使用Modelfile创建
    print(f"📝 使用Modelfile方式...")
    modelfile = create_modelfile(hf_model_name, local_name)
    
    cmd = f"ollama create {local_name} -f {modelfile}"
    success = run_command(cmd, f"创建模型 {local_name}")
    
    # 清理临时文件
    if os.path.exists(modelfile):
        os.remove(modelfile)
        print(f"🗑️ 清理临时文件: {modelfile}")
    
    return success

def main():
    """主函数"""
    print("🦙 Questions-Gen HuggingFace -> Ollama 导入工具")
    print("="*60)
    
    # 检查Ollama是否可用
    if not run_command("ollama --version", "检查Ollama版本"):
        print("❌ Ollama未安装或不可用!")
        return
    
    # 你的三个模型
    models = {
        "stage1": "xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged",
        "stage2": "xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit", 
        "final": "xingqiang/questions-gen-qwen3-14b-final-merged-16bit"
    }
    
    print(f"\n📋 发现 {len(models)} 个模型待导入:")
    for stage, hf_name in models.items():
        print(f"   {stage}: {hf_name}")
    
    # 询问用户
    choice = input(f"\n选择导入模式:\n1. 导入所有模型\n2. 导入final模型（推荐）\n3. 自定义选择\n请输入选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 导入所有模型
        print(f"\n🚀 导入所有模型...")
        success_count = 0
        for stage, hf_name in models.items():
            local_name = f"questions-gen-{stage}"
            if import_model_to_ollama(hf_name, local_name):
                success_count += 1
        
        print(f"\n📊 导入结果: {success_count}/{len(models)} 成功")
        
    elif choice == "2":
        # 只导入final模型
        print(f"\n🚀 导入final模型...")
        hf_name = models["final"]
        local_name = "questions-gen"
        success = import_model_to_ollama(hf_name, local_name)
        if success:
            print(f"✅ 成功导入! 使用方法: ollama run questions-gen")
        
    elif choice == "3":
        # 自定义选择
        print(f"\n🔧 自定义模式:")
        for stage, hf_name in models.items():
            if input(f"导入 {stage} 模型? (y/N): ").lower().startswith('y'):
                local_name = f"questions-gen-{stage}"
                import_model_to_ollama(hf_name, local_name)
    
    else:
        print("❌ 无效选择!")
        return
    
    # 显示最终结果
    print(f"\n📋 查看已导入的模型:")
    run_command("ollama list", "列出所有模型")
    
    print(f"\n💡 使用示例:")
    print("   ollama run questions-gen 'Generate a calculus problem:'")
    print("   ollama run questions-gen-final 'Create a geometry problem:'")

if __name__ == "__main__":
    main()
