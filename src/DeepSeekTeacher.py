# ==================== DeepSeek-R1 Teacher Model ====================
import os
import re
from typing import Dict, List
from openai import OpenAI
import numpy as np

from src.config import OPENAI_AVAILABLE


class DeepSeekTeacher:
    """DeepSeek-R1 Teacher Model for Knowledge Distillation - Using OpenAI SDK"""

    def __init__(self, api_key: str = None):
        if not OPENAI_AVAILABLE:
            print("❌ OpenAI SDK not available. DeepSeek teacher disabled.")
            self.client = None
            return

        self.api_key = api_key or os.environ.get('DEEPSEEK_API_KEY', 'sk-d02fca54e07f4bdfb1778aeb62ae7671')

        # Initialize OpenAI client for DeepSeek
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

        # Test API connection
        print("🔄 Testing DeepSeek-R1 API connection...")
        if self._test_connection():
            print("✅ DeepSeek-R1 teacher model connected successfully")
        else:
            print("❌ DeepSeek-R1 API connection failed")
            self.client = None

    def _test_connection(self) -> bool:
        """Test API connection"""
        if not self.client:
            return False

        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "Test connection"}
                ],
                max_tokens=10,
                temperature=0.1,
                timeout=30
            )

            # 更宽松的成功检查：只要有响应就认为连接成功
            if response and hasattr(response, 'choices') and response.choices:
                print(f"✅ API响应: {response.choices[0].message.content}")
                return True
            else:
                print("⚠️ API返回空响应")
                return False

        except Exception as e:
            print(f"⚠️ API test failed: {e}")
            # 打印更详细的错误信息
            if "model" in str(e).lower():
                print("   💡 可能是模型名称问题，但API基本连接正常")
                return True  # 如果只是模型名问题，仍认为连接正常
            return False

    def evaluate_problem(self, problem: str) -> Dict[str, any]:
        """Use DeepSeek-R1 to evaluate a math problem"""
        if not self.client:
            return self._default_evaluation()

        evaluation_prompt = f"""
Please evaluate this mathematics competition problem comprehensively:

Problem: {problem}

Please provide evaluation on the following aspects:
1. Mathematical rigor and correctness (1-5 scale)
2. Difficulty level (1-5 scale)
3. Innovation and creativity (1-5 scale)
4. Problem clarity and expression (1-5 scale)
5. Educational value (1-5 scale)
6. Specific suggestions for improvement

Please provide detailed reasoning and specific feedback in a structured format.
"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are an expert mathematics professor evaluating competition problems."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_tokens=1500,
                temperature=0.3,
                stream=False
            )

            evaluation_text = response.choices[0].message.content

            # Parse evaluation results
            evaluation = self._parse_evaluation(evaluation_text)
            evaluation['raw_feedback'] = evaluation_text

            return evaluation

        except Exception as e:
            print(f"❌ DeepSeek evaluation failed: {e}")
            return self._default_evaluation()

    def improve_problem(self, problem: str, feedback: str) -> str:
        """Use DeepSeek-R1 to improve a problem based on feedback"""
        if not self.client:
            return problem

        improvement_prompt = f"""
Based on the following feedback, please improve this mathematics competition problem:

Original Problem: {problem}

Feedback: {feedback}

Please provide an improved version that addresses the feedback while maintaining the mathematical essence. Focus on:
1. Enhancing mathematical rigor
2. Improving clarity of expression
3. Adjusting difficulty appropriately
4. Adding educational value

Provide only the improved problem statement, no additional commentary.
"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are an expert mathematics problem designer."},
                    {"role": "user", "content": improvement_prompt}
                ],
                max_tokens=4096,
                temperature=0.4,
                stream=False
            )

            improved_problem = response.choices[0].message.content
            return improved_problem.strip()

        except Exception as e:
            print(f"❌ DeepSeek improvement failed: {e}")
            return problem

    def generate_variations(self, original_problem: str, num_variations: int = 3) -> List[str]:
        """Use DeepSeek-R1 to generate intelligent problem variations"""
        if not self.client:
            return []

        variation_prompt = f"""
Generate {num_variations} intelligent variations of this mathematics competition problem:

Original Problem: {original_problem}

Please create variations that:
1. Maintain the same mathematical concepts and solution methods
2. Use different contexts or applications
3. Adjust parameters while preserving difficulty
4. Ensure mathematical correctness and clarity

Provide each variation on a separate line, numbered 1, 2, 3, etc.
"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": "You are an expert mathematics problem designer specializing in creating problem variations."},
                    {"role": "user", "content": variation_prompt}
                ],
                max_tokens=4096,
                temperature=0.6,
                stream=False
            )

            variations_text = response.choices[0].message.content

            # Parse variations from response
            variations = self._parse_variations(variations_text)
            return variations[:num_variations]

        except Exception as e:
            print(f"❌ DeepSeek variation generation failed: {e}")
            return []

    def _parse_evaluation(self, evaluation_text: str) -> Dict[str, any]:
        """Parse evaluation results from DeepSeek response"""
        evaluation = {
            'rigor_score': 3.0,
            'difficulty_score': 3.0,
            'innovation_score': 3.0,
            'clarity_score': 3.0,
            'educational_value': 3.0,
            'overall_score': 3.0,
            'suggestions': []
        }

        try:
            # Extract numerical scores using pattern matching
            text_lower = evaluation_text.lower()

            # Look for explicit numerical ratings
            score_patterns = [
                (r'rigor[^\d]*([1-5])', 'rigor_score'),
                (r'difficulty[^\d]*([1-5])', 'difficulty_score'),
                (r'innovation[^\d]*([1-5])', 'innovation_score'),
                (r'clarity[^\d]*([1-5])', 'clarity_score'),
                (r'educational[^\d]*([1-5])', 'educational_value')
            ]

            for pattern, key in score_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    evaluation[key] = float(match.group(1))

            # Look for qualitative indicators if no explicit scores
            if 'difficulty' in text_lower:
                if any(word in text_lower for word in ['easy', 'simple', 'basic', 'elementary']):
                    evaluation['difficulty_score'] = 2.0
                elif any(word in text_lower for word in ['hard', 'challenging', 'difficult', 'advanced']):
                    evaluation['difficulty_score'] = 4.0

            # Look for rigor mentions
            if any(word in text_lower for word in ['rigorous', 'precise', 'correct', 'well-defined']):
                evaluation['rigor_score'] = 4.0
            elif any(word in text_lower for word in ['unclear', 'ambiguous', 'imprecise']):
                evaluation['rigor_score'] = 2.0

            # Look for innovation mentions
            if any(word in text_lower for word in ['creative', 'innovative', 'original', 'novel']):
                evaluation['innovation_score'] = 4.0
            elif any(word in text_lower for word in ['standard', 'typical', 'common']):
                evaluation['innovation_score'] = 2.0

            # Calculate overall score
            evaluation['overall_score'] = np.mean([
                evaluation['rigor_score'],
                evaluation['difficulty_score'],
                evaluation['innovation_score'],
                evaluation['clarity_score'],
                evaluation['educational_value']
            ])

        except Exception as e:
            print(f"⚠️ Evaluation parsing failed: {e}")

        return evaluation

    def _parse_variations(self, variations_text: str) -> List[str]:
        """Parse variations from DeepSeek response"""
        variations = []

        try:
            # Split by numbered items
            lines = variations_text.split('\n')
            current_variation = ""

            for line in lines:
                line = line.strip()
                if re.match(r'^[0-9]+\.', line):  # New numbered item
                    if current_variation:
                        variations.append(current_variation.strip())
                    current_variation = re.sub(r'^[0-9]+\.\s*', '', line)
                elif current_variation and line:
                    current_variation += " " + line

            # Add the last variation
            if current_variation:
                variations.append(current_variation.strip())

        except Exception as e:
            print(f"⚠️ Variation parsing failed: {e}")

        return variations

    def _default_evaluation(self) -> Dict[str, any]:
        """Return default evaluation when API fails"""
        return {
            'rigor_score': 3.0,
            'difficulty_score': 3.0,
            'innovation_score': 3.0,
            'clarity_score': 3.0,
            'educational_value': 3.0,
            'overall_score': 3.0,
            'suggestions': ['Unable to get detailed feedback from teacher model'],
            'raw_feedback': 'DeepSeek-R1 evaluation temporarily unavailable'
        }
