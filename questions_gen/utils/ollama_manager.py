# -*- coding: utf-8 -*-
"""Ollama integration for Questions-Gen models."""

import os
import json
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional
from pathlib import Path


class OllamaManager:
    """管理Ollama模型的推送和部署。"""

    def __init__(self):
        self.ollama_available = self._check_ollama_installation()
        self.models_info = {
            "stage1": {
                "hf_name": "xingqiang/QuestionsGen-Qwen3-14b-stage1-fp-merged",
                "ollama_name": "questions-gen-stage1",
                "description": "Questions-Gen Stage 1: Basic mathematical problem generation"
            },
            "stage2": {
                "hf_name": "xingqiang/questions-gen-qwen3-14b-stage2-merged-16bit", 
                "ollama_name": "questions-gen-stage2",
                "description": "Questions-Gen Stage 2: GRPO optimized with variation generation"
            },
            "final": {
                "hf_name": "xingqiang/questions-gen-qwen3-14b-final-merged-16bit",
                "ollama_name": "questions-gen-final",
                "description": "Questions-Gen Final: Complete pipeline with knowledge distillation"
            }
        }

    def _check_ollama_installation(self) -> bool:
        """检查Ollama是否已安装。"""
        try:
            result = subprocess.run(['ollama', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ Ollama detected: {result.stdout.strip()}")
                return True
            else:
                print("❌ Ollama not found")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Ollama not installed or not in PATH")
            return False

    def create_modelfile(self, model_stage: str, custom_template: Optional[str] = None) -> str:
        """创建Ollama Modelfile。"""
        if model_stage not in self.models_info:
            raise ValueError(f"Unknown model stage: {model_stage}")

        model_info = self.models_info[model_stage]
        
        # 默认的系统提示
        default_template = """You are Questions-Gen, an expert mathematical problem generator. You create high-quality competition-level mathematics problems across various domains including algebra, geometry, calculus, number theory, combinatorics, and analysis.

Key capabilities:
- Generate original, challenging mathematical problems
- Create problem variations with different contexts
- Provide step-by-step solutions when requested
- Adapt difficulty levels from beginner to expert
- Focus on educational value and mathematical rigor

Always respond with well-structured problems that include clear problem statements and, when appropriate, solution hints or complete solutions."""

        template = custom_template or default_template

        modelfile_content = f'''FROM {model_info["hf_name"]}

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
{{{{ .Response }}}}<|im_end|>
"""

SYSTEM """{template}"""

PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"

# Model metadata
PARAMETER num_ctx 4096
PARAMETER num_predict 512
'''

        return modelfile_content

    def push_model_to_ollama(self, model_stage: str, 
                           custom_template: Optional[str] = None,
                           force: bool = False) -> bool:
        """将模型推送到Ollama。"""
        if not self.ollama_available:
            print("❌ Ollama not available. Please install Ollama first.")
            return False

        if model_stage not in self.models_info:
            print(f"❌ Unknown model stage: {model_stage}")
            return False

        model_info = self.models_info[model_stage]
        ollama_name = model_info["ollama_name"]

        print(f"🚀 Pushing {model_stage} model to Ollama as '{ollama_name}'")
        print(f"📂 Source: {model_info['hf_name']}")

        # Check if model already exists
        if not force and self._model_exists_in_ollama(ollama_name):
            print(f"⚠️ Model '{ollama_name}' already exists in Ollama.")
            print("Use force=True to overwrite, or delete the existing model first.")
            return False

        # Create temporary Modelfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.modelfile', delete=False) as f:
            modelfile_content = self.create_modelfile(model_stage, custom_template)
            f.write(modelfile_content)
            modelfile_path = f.name

        try:
            print(f"📝 Created Modelfile: {modelfile_path}")
            print("🔄 Creating Ollama model (this may take several minutes)...")

            # Run ollama create command
            cmd = ['ollama', 'create', ollama_name, '-f', modelfile_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout

            if result.returncode == 0:
                print(f"✅ Successfully created Ollama model: {ollama_name}")
                print(f"📊 Model size and details:")
                self._show_model_info(ollama_name)
                return True
            else:
                print(f"❌ Failed to create Ollama model")
                print(f"Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("❌ Model creation timed out (30 minutes)")
            return False
        except Exception as e:
            print(f"❌ Error during model creation: {e}")
            return False
        finally:
            # Clean up temporary file
            try:
                os.unlink(modelfile_path)
            except:
                pass

    def push_all_models(self, custom_templates: Optional[Dict[str, str]] = None,
                       force: bool = False) -> Dict[str, bool]:
        """推送所有模型到Ollama。"""
        print("🚀 Pushing all Questions-Gen models to Ollama...")
        print("="*60)

        results = {}
        custom_templates = custom_templates or {}

        for stage in self.models_info.keys():
            print(f"\n📦 Processing {stage} model...")
            template = custom_templates.get(stage)
            success = self.push_model_to_ollama(stage, template, force)
            results[stage] = success

            if success:
                print(f"✅ {stage} model pushed successfully")
            else:
                print(f"❌ {stage} model push failed")

        # Summary
        successful = sum(results.values())
        total = len(results)
        print(f"\n📊 Push Summary: {successful}/{total} models successful")

        if successful > 0:
            print(f"\n🎉 Available Ollama models:")
            self.list_questions_gen_models()

        return results

    def _model_exists_in_ollama(self, model_name: str) -> bool:
        """检查模型是否已存在于Ollama中。"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            return model_name in result.stdout
        except:
            return False

    def _show_model_info(self, model_name: str):
        """显示模型信息。"""
        try:
            result = subprocess.run(['ollama', 'show', model_name], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(result.stdout)
        except:
            print("Could not retrieve model info")

    def list_questions_gen_models(self) -> List[str]:
        """列出所有Questions-Gen相关的Ollama模型。"""
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                qgen_models = []
                
                print("📋 Questions-Gen models in Ollama:")
                for line in lines[1:]:  # Skip header
                    if 'questions-gen' in line.lower():
                        model_name = line.split()[0]
                        qgen_models.append(model_name)
                        print(f"   ✅ {model_name}")
                
                if not qgen_models:
                    print("   No Questions-Gen models found")
                
                return qgen_models
            else:
                print("❌ Failed to list Ollama models")
                return []
        except Exception as e:
            print(f"❌ Error listing models: {e}")
            return []

    def test_model_in_ollama(self, model_name: str, 
                            test_prompts: Optional[List[str]] = None) -> Dict:
        """在Ollama中测试模型。"""
        if test_prompts is None:
            test_prompts = [
                "Generate a calculus competition problem:",
                "Create an algebra problem with moderate difficulty:",
                "Design a geometry proof problem:"
            ]

        print(f"🧪 Testing model '{model_name}' in Ollama...")
        test_results = {
            "model_name": model_name,
            "test_results": [],
            "success_rate": 0.0
        }

        successful_tests = 0

        for i, prompt in enumerate(test_prompts):
            print(f"\n📝 Test {i+1}/{len(test_prompts)}: {prompt[:50]}...")
            
            try:
                # Use ollama run command
                cmd = ['ollama', 'run', model_name, prompt]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    response = result.stdout.strip()
                    print(f"✅ Response: {response[:100]}...")
                    successful_tests += 1
                    
                    test_result = {
                        "prompt": prompt,
                        "response": response,
                        "success": True,
                        "error": None
                    }
                else:
                    print(f"❌ Error: {result.stderr}")
                    test_result = {
                        "prompt": prompt,
                        "response": None,
                        "success": False,
                        "error": result.stderr
                    }
                
                test_results["test_results"].append(test_result)

            except subprocess.TimeoutExpired:
                print("❌ Test timed out")
                test_result = {
                    "prompt": prompt,
                    "response": None,
                    "success": False,
                    "error": "Timeout"
                }
                test_results["test_results"].append(test_result)

            except Exception as e:
                print(f"❌ Test failed: {e}")
                test_result = {
                    "prompt": prompt,
                    "response": None,
                    "success": False,
                    "error": str(e)
                }
                test_results["test_results"].append(test_result)

        test_results["success_rate"] = successful_tests / len(test_prompts)
        
        print(f"\n📊 Test Summary: {successful_tests}/{len(test_prompts)} successful")
        print(f"🎯 Success Rate: {test_results['success_rate']:.1%}")

        return test_results

    def remove_model_from_ollama(self, model_name: str) -> bool:
        """从Ollama中删除模型。"""
        try:
            print(f"🗑️ Removing model '{model_name}' from Ollama...")
            result = subprocess.run(['ollama', 'rm', model_name], 
                                  capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print(f"✅ Successfully removed model: {model_name}")
                return True
            else:
                print(f"❌ Failed to remove model: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error removing model: {e}")
            return False

    def update_model_in_ollama(self, model_stage: str,
                              custom_template: Optional[str] = None) -> bool:
        """更新Ollama中的模型。"""
        if model_stage not in self.models_info:
            print(f"❌ Unknown model stage: {model_stage}")
            return False

        ollama_name = self.models_info[model_stage]["ollama_name"]
        
        print(f"🔄 Updating model '{ollama_name}' in Ollama...")
        
        # Remove existing model
        if self._model_exists_in_ollama(ollama_name):
            if not self.remove_model_from_ollama(ollama_name):
                print("❌ Failed to remove existing model")
                return False

        # Push new version
        return self.push_model_to_ollama(model_stage, custom_template, force=True)

    def generate_ollama_usage_guide(self) -> str:
        """生成Ollama使用指南。"""
        guide = """
# Questions-Gen Ollama Usage Guide

## Available Models

"""
        
        for stage, info in self.models_info.items():
            guide += f"### {info['ollama_name']}\n"
            guide += f"- **Stage**: {stage}\n"
            guide += f"- **Description**: {info['description']}\n"
            guide += f"- **Usage**: `ollama run {info['ollama_name']}`\n\n"

        guide += """
## Basic Usage Examples

### Generate a Competition Problem
```bash
ollama run questions-gen-final "Generate a challenging calculus competition problem:"
```

### Create Problem Variations
```bash
ollama run questions-gen-stage2 "Create a variation of this problem: Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1"
```

### Get Step-by-Step Solutions
```bash
ollama run questions-gen-final "Solve this step-by-step: Integrate ∫(2x³ - 3x² + x - 1)dx"
```

## Model Comparison

- **questions-gen-stage1**: Basic problem generation, good for simple math problems
- **questions-gen-stage2**: Enhanced with variation generation capabilities  
- **questions-gen-final**: Most advanced, includes knowledge distillation from expert models

## Tips for Best Results

1. Be specific about the type of problem you want
2. Specify difficulty level (e.g., "beginner", "advanced", "competition-level")
3. Include context when requesting variations
4. Ask for solutions separately if needed

## API Usage

You can also use the Ollama API:

```python
import requests

def generate_problem(prompt):
    response = requests.post('http://localhost:11434/api/generate',
                           json={
                               'model': 'questions-gen-final',
                               'prompt': prompt,
                               'stream': False
                           })
    return response.json()['response']

problem = generate_problem("Create a number theory competition problem:")
print(problem)
```
"""

        return guide

    def save_usage_guide(self, filename: str = "ollama_usage_guide.md") -> str:
        """保存使用指南到文件。"""
        guide_content = self.generate_ollama_usage_guide()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(guide_content)
        
        print(f"📚 Usage guide saved to: {filename}")
        return filename
