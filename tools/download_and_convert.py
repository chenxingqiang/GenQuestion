#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载HuggingFace模型并转换为Ollama兼容格式的脚本
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

def run_command(cmd, description="", cwd=None):
    """运行命令并返回结果"""
    print(f"🔄 {description}")
    print(f"   Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
        if result.returncode == 0:
            print(f"✅ {description} 成功!")
            if result.stdout.strip():
                print(f"   输出: {result.stdout.strip()}")
            return True, result.stdout
        else:
            print(f"❌ {description} 失败!")
            if result.stderr.strip():
                print(f"   错误: {result.stderr.strip()}")
            return False, result.stderr
    except Exception as e:
        print(f"❌ 执行命令时出错: {e}")
        return False, str(e)

def check_dependencies():
    """检查必要的依赖"""
    print("🔍 检查依赖...")
    
    # 检查git-lfs
    success, _ = run_command("git lfs version", "检查git-lfs")
    if not success:
        print("❌ git-lfs未安装!")
        print("💡 安装方法: brew install git-lfs")
        return False
    
    # 检查huggingface-cli
    success, _ = run_command("pip show huggingface_hub", "检查huggingface_hub")
    if not success:
        print("⚠️ huggingface_hub未安装，尝试安装...")
        success, _ = run_command("pip install --user huggingface_hub", "安装huggingface_hub")
        if not success:
            print("❌ 无法安装huggingface_hub")
            return False
    
    return True

def download_model_from_hf(model_name, local_path):
    """从HuggingFace下载模型"""
    print(f"\n📥 下载模型: {model_name}")
    print(f"   目标路径: {local_path}")
    
    # 确保目录存在
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    # 使用git clone下载
    repo_url = f"https://huggingface.co/{model_name}"
    cmd = f"git clone {repo_url} {local_path}"
    
    success, output = run_command(cmd, f"克隆模型仓库 {model_name}")
    
    if success:
        # 下载LFS文件
        success, _ = run_command("git lfs pull", f"下载大文件", cwd=local_path)
    
    return success

def create_ollama_modelfile(model_path, model_name):
    """创建Ollama Modelfile"""
    modelfile_content = f"""FROM {model_path}

TEMPLATE \"\"\"<|im_start|>system
You are a helpful assistant that generates high-quality mathematical problems. Please generate clear, well-structured problems with appropriate difficulty levels.
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

SYSTEM \"\"\"You are an expert mathematics problem generator. Generate educational and challenging mathematical problems across various topics including algebra, geometry, calculus, and statistics.\"\"\"
"""
    
    filename = f"{model_name}.modelfile"
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"📝 创建Modelfile: {filename}")
    return filename

def import_to_ollama(model_path, ollama_name):
    """导入模型到Ollama"""
    print(f"\n🦙 导入到Ollama: {ollama_name}")
    
    # 创建Modelfile
    modelfile = create_ollama_modelfile(model_path, ollama_name)
    
    try:
        # 创建Ollama模型
        cmd = f"ollama create {ollama_name} -f {modelfile}"
        success, output = run_command(cmd, f"创建Ollama模型 {ollama_name}")
        
        if success:
            print(f"✅ 成功创建Ollama模型: {ollama_name}")
            print(f"💡 使用方法: ollama run {ollama_name}")
            return True
        else:
            print(f"❌ 创建Ollama模型失败")
            return False
            
    finally:
        # 清理Modelfile
        if os.path.exists(modelfile):
            os.remove(modelfile)
            print(f"🗑️ 清理临时文件: {modelfile}")

def main():
    """主函数"""
    print("🚀 HuggingFace -> Ollama 模型导入工具")
    print("="*60)
    
    # 检查依赖
    if not check_dependencies():
        print("❌ 依赖检查失败!")
        return
    
    # 你的模型列表
    models = {
        "final": "xingqiang/questions-gen-qwen3-14b-final-merged-16bit",
        "stage2": "xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit", 
        "stage1": "xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged"
    }
    
    print(f"\n📋 可用模型:")
    for i, (stage, model_name) in enumerate(models.items(), 1):
        print(f"   {i}. {stage}: {model_name}")
    
    # 选择模型
    choice = input(f"\n选择要导入的模型:\n1. final模型 (推荐)\n2. 所有模型\n3. 自定义选择\n请输入 (1/2/3): ").strip()
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="ollama_import_")
    print(f"📁 临时目录: {temp_dir}")
    
    try:
        if choice == "1":
            # 只下载final模型
            model_name = models["final"]
            local_path = os.path.join(temp_dir, "final")
            
            if download_model_from_hf(model_name, local_path):
                import_to_ollama(local_path, "questions-gen")
                
        elif choice == "2":
            # 下载所有模型
            for stage, model_name in models.items():
                local_path = os.path.join(temp_dir, stage)
                
                if download_model_from_hf(model_name, local_path):
                    ollama_name = f"questions-gen-{stage}"
                    import_to_ollama(local_path, ollama_name)
                else:
                    print(f"❌ 跳过 {stage} 模型")
                    
        elif choice == "3":
            # 自定义选择
            for stage, model_name in models.items():
                if input(f"下载 {stage} 模型? (y/N): ").lower().startswith('y'):
                    local_path = os.path.join(temp_dir, stage)
                    
                    if download_model_from_hf(model_name, local_path):
                        ollama_name = f"questions-gen-{stage}"
                        import_to_ollama(local_path, ollama_name)
        else:
            print("❌ 无效选择!")
            return
            
    finally:
        # 清理临时目录
        print(f"\n🗑️ 清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # 显示结果
    print(f"\n📋 查看导入的模型:")
    run_command("ollama list | grep questions-gen", "查看Questions-Gen模型")
    
    print(f"\n🎉 导入完成!")
    print(f"💡 使用示例:")
    print(f"   ollama run questions-gen 'Generate a calculus problem'")
    print(f"   ollama run questions-gen-final 'Create a geometry problem'")

if __name__ == "__main__":
    main()
