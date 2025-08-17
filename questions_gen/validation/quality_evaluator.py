# -*- coding: utf-8 -*-
"""Advanced quality evaluation for generated questions."""

import re
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ..models.reward_calculator import RewardCalculator
from ..models.deepseek_teacher import DeepSeekTeacher


class QualityEvaluator:
    """Advanced quality evaluation system for generated mathematical questions."""

    def __init__(self):
        self.reward_calculator = RewardCalculator()
        self.deepseek_teacher = DeepSeekTeacher()
        
        # Mathematical content patterns
        self.math_patterns = {
            "equations": r'[a-z]\s*[=<>≤≥≠]\s*[^a-z\s]',
            "functions": r'f\([^)]+\)|g\([^)]+\)|h\([^)]+\)',
            "derivatives": r"f'|f''|df/dx|∂|derivative",
            "integrals": r'∫|integral|integrate',
            "summations": r'∑|sum|Σ',
            "limits": r'lim|limit|→|approaches',
            "inequalities": r'[<>≤≥≠]',
            "fractions": r'\d+/\d+|\frac{',
            "exponents": r'\^|\*\*|squared|cubed',
            "roots": r'√|sqrt|root',
            "trigonometry": r'sin|cos|tan|sec|csc|cot',
            "logarithms": r'log|ln|exp',
            "sets": r'[∈∉⊂⊆∪∩]|element|subset|union|intersection',
            "geometry": r'triangle|circle|angle|area|perimeter|volume',
            "probability": r'probability|random|chance|odds',
            "statistics": r'mean|median|mode|variance|standard deviation'
        }
        
        # Quality criteria weights
        self.quality_weights = {
            "mathematical_content": 0.25,
            "clarity": 0.20,
            "difficulty": 0.20,
            "completeness": 0.15,
            "originality": 0.10,
            "educational_value": 0.10
        }

    def comprehensive_evaluation(self, question: str, context: Optional[Dict] = None) -> Dict:
        """进行全面的质量评估。"""
        evaluation = {
            "question": question,
            "timestamp": time.time(),
            "metrics": {},
            "overall_score": 0.0,
            "grade": "F",
            "recommendations": []
        }

        # 1. Mathematical Content Analysis
        math_analysis = self._analyze_mathematical_content(question)
        evaluation["metrics"]["mathematical_content"] = math_analysis

        # 2. Clarity Assessment
        clarity_score = self._assess_clarity(question)
        evaluation["metrics"]["clarity"] = clarity_score

        # 3. Difficulty Estimation
        difficulty_analysis = self._estimate_difficulty(question)
        evaluation["metrics"]["difficulty"] = difficulty_analysis

        # 4. Completeness Check
        completeness_score = self._check_completeness(question)
        evaluation["metrics"]["completeness"] = completeness_score

        # 5. Originality Assessment
        originality_score = self._assess_originality(question, context)
        evaluation["metrics"]["originality"] = originality_score

        # 6. Educational Value
        educational_score = self._assess_educational_value(question)
        evaluation["metrics"]["educational_value"] = educational_score

        # 7. Teacher Model Evaluation (if available)
        if self.deepseek_teacher.client:
            try:
                teacher_eval = self.deepseek_teacher.evaluate_problem(question)
                evaluation["metrics"]["teacher_evaluation"] = teacher_eval
            except Exception as e:
                evaluation["metrics"]["teacher_evaluation"] = {"error": str(e)}

        # Calculate overall score
        overall_score = self._calculate_overall_score(evaluation["metrics"])
        evaluation["overall_score"] = overall_score
        evaluation["grade"] = self._assign_grade(overall_score)

        # Generate recommendations
        evaluation["recommendations"] = self._generate_recommendations(evaluation["metrics"])

        return evaluation

    def _analyze_mathematical_content(self, question: str) -> Dict:
        """分析数学内容的丰富程度。"""
        content_analysis = {
            "patterns_found": {},
            "content_density": 0.0,
            "mathematical_concepts": [],
            "score": 0.0
        }

        text_lower = question.lower()
        total_patterns = 0

        # Check for mathematical patterns
        for pattern_name, pattern in self.math_patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                content_analysis["patterns_found"][pattern_name] = len(matches)
                content_analysis["mathematical_concepts"].append(pattern_name)
                total_patterns += len(matches)

        # Calculate content density
        word_count = len(question.split())
        content_analysis["content_density"] = total_patterns / word_count if word_count > 0 else 0

        # Calculate score based on diversity and density
        concept_diversity = len(content_analysis["mathematical_concepts"])
        max_expected_concepts = 5  # Reasonable expectation
        
        diversity_score = min(concept_diversity / max_expected_concepts, 1.0)
        density_score = min(content_analysis["content_density"] * 10, 1.0)  # Scale density
        
        content_analysis["score"] = (diversity_score + density_score) / 2

        return content_analysis

    def _assess_clarity(self, question: str) -> Dict:
        """评估问题的清晰度。"""
        clarity_metrics = {
            "readability": 0.0,
            "structure": 0.0,
            "ambiguity": 0.0,
            "score": 0.0
        }

        # Basic readability (sentence structure)
        sentences = question.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Optimal sentence length for math problems: 10-20 words
        if 10 <= avg_sentence_length <= 20:
            clarity_metrics["readability"] = 1.0
        elif avg_sentence_length < 10:
            clarity_metrics["readability"] = avg_sentence_length / 10
        else:
            clarity_metrics["readability"] = max(0, 1 - (avg_sentence_length - 20) / 20)

        # Structure assessment (presence of clear problem statement)
        structure_indicators = [
            'find', 'prove', 'show', 'determine', 'calculate', 'solve',
            'evaluate', 'verify', 'demonstrate', 'compute'
        ]
        
        structure_score = sum(1 for indicator in structure_indicators 
                            if indicator in question.lower()) / len(structure_indicators)
        clarity_metrics["structure"] = min(structure_score * 2, 1.0)

        # Ambiguity assessment (negative indicators)
        ambiguity_indicators = [
            'maybe', 'perhaps', 'possibly', 'might', 'could be',
            'unclear', 'ambiguous', 'confusing'
        ]
        
        ambiguity_count = sum(1 for indicator in ambiguity_indicators 
                            if indicator in question.lower())
        clarity_metrics["ambiguity"] = max(0, 1 - ambiguity_count * 0.3)

        # Overall clarity score
        clarity_metrics["score"] = np.mean([
            clarity_metrics["readability"],
            clarity_metrics["structure"], 
            clarity_metrics["ambiguity"]
        ])

        return clarity_metrics

    def _estimate_difficulty(self, question: str) -> Dict:
        """估算问题难度。"""
        difficulty_analysis = {
            "lexical_complexity": 0.0,
            "mathematical_complexity": 0.0,
            "problem_type": "unknown",
            "estimated_level": "beginner",
            "score": 0.0
        }

        text_lower = question.lower()

        # Lexical complexity (word length and sophistication)
        words = question.split()
        avg_word_length = np.mean([len(word) for word in words])
        difficulty_analysis["lexical_complexity"] = min(avg_word_length / 8, 1.0)

        # Mathematical complexity
        complexity_indicators = {
            "basic": ["add", "subtract", "multiply", "divide", "sum", "difference"],
            "intermediate": ["equation", "function", "graph", "solve", "variable"],
            "advanced": ["derivative", "integral", "limit", "theorem", "proof", "analysis"],
            "expert": ["topology", "manifold", "homomorphism", "isomorphism", "abstract"]
        }

        max_level_score = 0
        detected_level = "basic"

        for level, indicators in complexity_indicators.items():
            level_score = sum(1 for indicator in indicators if indicator in text_lower)
            if level_score > 0:
                if level == "basic":
                    max_level_score = max(max_level_score, 0.25)
                elif level == "intermediate":
                    max_level_score = max(max_level_score, 0.5)
                    detected_level = "intermediate"
                elif level == "advanced":
                    max_level_score = max(max_level_score, 0.75)
                    detected_level = "advanced"
                elif level == "expert":
                    max_level_score = max(max_level_score, 1.0)
                    detected_level = "expert"

        difficulty_analysis["mathematical_complexity"] = max_level_score
        difficulty_analysis["estimated_level"] = detected_level

        # Overall difficulty score
        difficulty_analysis["score"] = (
            difficulty_analysis["lexical_complexity"] * 0.3 +
            difficulty_analysis["mathematical_complexity"] * 0.7
        )

        return difficulty_analysis

    def _check_completeness(self, question: str) -> Dict:
        """检查问题的完整性。"""
        completeness_metrics = {
            "has_clear_question": False,
            "has_context": False,
            "has_constraints": False,
            "has_solution_hint": False,
            "score": 0.0
        }

        text_lower = question.lower()

        # Check for clear question
        question_indicators = ['?', 'find', 'prove', 'show', 'determine', 'calculate']
        completeness_metrics["has_clear_question"] = any(
            indicator in text_lower for indicator in question_indicators
        )

        # Check for context/setup
        context_indicators = ['given', 'let', 'consider', 'suppose', 'assume']
        completeness_metrics["has_context"] = any(
            indicator in text_lower for indicator in context_indicators
        )

        # Check for constraints
        constraint_indicators = ['where', 'such that', 'if', 'when', 'provided']
        completeness_metrics["has_constraints"] = any(
            indicator in text_lower for indicator in constraint_indicators
        )

        # Check for solution hints
        hint_indicators = ['hint', 'solution', 'answer', 'result', 'therefore']
        completeness_metrics["has_solution_hint"] = any(
            indicator in text_lower for indicator in hint_indicators
        )

        # Calculate completeness score
        components = [
            completeness_metrics["has_clear_question"],
            completeness_metrics["has_context"],
            completeness_metrics["has_constraints"],
            completeness_metrics["has_solution_hint"]
        ]
        
        completeness_metrics["score"] = sum(components) / len(components)

        return completeness_metrics

    def _assess_originality(self, question: str, context: Optional[Dict] = None) -> Dict:
        """评估问题的原创性。"""
        originality_metrics = {
            "uniqueness": 0.0,
            "creativity_indicators": [],
            "innovation_score": 0.0,
            "score": 0.0
        }

        # Check for creativity indicators
        creativity_patterns = [
            "real.world", "application", "practical", "innovative", "novel",
            "creative", "unique", "original", "interdisciplinary"
        ]

        text_lower = question.lower()
        found_indicators = [pattern for pattern in creativity_patterns 
                          if pattern.replace('.', ' ') in text_lower]
        
        originality_metrics["creativity_indicators"] = found_indicators
        creativity_score = min(len(found_indicators) / 3, 1.0)

        # Innovation score based on problem structure
        innovation_indicators = [
            "combine", "integrate", "synthesize", "connect", "relate",
            "compare", "contrast", "analyze", "evaluate"
        ]
        
        innovation_count = sum(1 for indicator in innovation_indicators 
                             if indicator in text_lower)
        innovation_score = min(innovation_count / 3, 1.0)

        originality_metrics["innovation_score"] = innovation_score

        # If context provided, check similarity with known problems
        if context and "historical_questions" in context:
            similarity_scores = []
            for hist_q in context["historical_questions"][-10:]:  # Check last 10
                try:
                    vectorizer = TfidfVectorizer(max_features=100)
                    tfidf_matrix = vectorizer.fit_transform([question, hist_q])
                    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                    similarity_scores.append(similarity)
                except:
                    pass
            
            if similarity_scores:
                max_similarity = max(similarity_scores)
                uniqueness = 1 - max_similarity
                originality_metrics["uniqueness"] = max(0, uniqueness)
            else:
                originality_metrics["uniqueness"] = 0.8  # Default high uniqueness
        else:
            originality_metrics["uniqueness"] = 0.7  # Default moderate uniqueness

        # Overall originality score
        originality_metrics["score"] = np.mean([
            originality_metrics["uniqueness"],
            creativity_score,
            innovation_score
        ])

        return originality_metrics

    def _assess_educational_value(self, question: str) -> Dict:
        """评估教育价值。"""
        educational_metrics = {
            "learning_objectives": [],
            "skill_development": 0.0,
            "pedagogical_quality": 0.0,
            "score": 0.0
        }

        text_lower = question.lower()

        # Learning objectives
        learning_patterns = {
            "problem_solving": ["solve", "find", "determine", "calculate"],
            "proof_writing": ["prove", "show", "demonstrate", "verify"],
            "conceptual_understanding": ["explain", "why", "how", "relationship"],
            "application": ["apply", "use", "implement", "practical"],
            "analysis": ["analyze", "evaluate", "compare", "assess"]
        }

        detected_objectives = []
        for objective, patterns in learning_patterns.items():
            if any(pattern in text_lower for pattern in patterns):
                detected_objectives.append(objective)

        educational_metrics["learning_objectives"] = detected_objectives
        skill_score = len(detected_objectives) / len(learning_patterns)

        # Pedagogical quality
        pedagogical_indicators = [
            "step by step", "gradually", "systematically", "method",
            "approach", "strategy", "technique", "process"
        ]

        pedagogical_count = sum(1 for indicator in pedagogical_indicators 
                              if indicator in text_lower)
        pedagogical_quality = min(pedagogical_count / 3, 1.0)

        educational_metrics["skill_development"] = skill_score
        educational_metrics["pedagogical_quality"] = pedagogical_quality

        # Overall educational value
        educational_metrics["score"] = (skill_score + pedagogical_quality) / 2

        return educational_metrics

    def _calculate_overall_score(self, metrics: Dict) -> float:
        """计算总体质量分数。"""
        component_scores = {}
        
        # Extract scores from metrics
        if "mathematical_content" in metrics:
            component_scores["mathematical_content"] = metrics["mathematical_content"]["score"]
        
        if "clarity" in metrics:
            component_scores["clarity"] = metrics["clarity"]["score"]
            
        if "difficulty" in metrics:
            component_scores["difficulty"] = metrics["difficulty"]["score"]
            
        if "completeness" in metrics:
            component_scores["completeness"] = metrics["completeness"]["score"]
            
        if "originality" in metrics:
            component_scores["originality"] = metrics["originality"]["score"]
            
        if "educational_value" in metrics:
            component_scores["educational_value"] = metrics["educational_value"]["score"]

        # Calculate weighted average
        total_score = 0
        total_weight = 0
        
        for component, score in component_scores.items():
            if component in self.quality_weights:
                weight = self.quality_weights[component]
                total_score += score * weight
                total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0

    def _assign_grade(self, score: float) -> str:
        """根据分数分配等级。"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.85:
            return "A"
        elif score >= 0.8:
            return "A-"
        elif score >= 0.75:
            return "B+"
        elif score >= 0.7:
            return "B"
        elif score >= 0.65:
            return "B-"
        elif score >= 0.6:
            return "C+"
        elif score >= 0.55:
            return "C"
        elif score >= 0.5:
            return "C-"
        elif score >= 0.4:
            return "D"
        else:
            return "F"

    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """生成改进建议。"""
        recommendations = []

        # Mathematical content recommendations
        if "mathematical_content" in metrics:
            math_score = metrics["mathematical_content"]["score"]
            if math_score < 0.6:
                recommendations.append("增加数学概念的多样性和深度")
                recommendations.append("包含更多数学符号和表达式")

        # Clarity recommendations
        if "clarity" in metrics:
            clarity_score = metrics["clarity"]["score"]
            if clarity_score < 0.7:
                recommendations.append("改善问题表述的清晰度")
                recommendations.append("使用更简洁明确的语言")

        # Difficulty recommendations
        if "difficulty" in metrics:
            diff_score = metrics["difficulty"]["score"]
            if diff_score < 0.4:
                recommendations.append("增加问题的数学复杂度")
            elif diff_score > 0.8:
                recommendations.append("适当降低问题难度以提高可解性")

        # Completeness recommendations
        if "completeness" in metrics:
            comp_score = metrics["completeness"]["score"]
            if comp_score < 0.7:
                recommendations.append("提供更完整的问题背景和约束条件")
                recommendations.append("确保问题陈述完整且无歧义")

        # Originality recommendations
        if "originality" in metrics:
            orig_score = metrics["originality"]["score"]
            if orig_score < 0.6:
                recommendations.append("增加问题的原创性和创新性")
                recommendations.append("考虑结合实际应用场景")

        # Educational value recommendations
        if "educational_value" in metrics:
            edu_score = metrics["educational_value"]["score"]
            if edu_score < 0.6:
                recommendations.append("明确学习目标和技能发展重点")
                recommendations.append("增强问题的教学价值")

        return recommendations

    def batch_evaluate_questions(self, questions: List[str], 
                                context: Optional[Dict] = None) -> Dict:
        """批量评估多个问题。"""
        import time
        
        batch_results = {
            "timestamp": time.time(),
            "total_questions": len(questions),
            "evaluations": [],
            "summary_statistics": {}
        }

        print(f"🔍 Evaluating {len(questions)} questions...")

        # Evaluate each question
        for i, question in enumerate(questions):
            print(f"   Question {i+1}/{len(questions)}")
            evaluation = self.comprehensive_evaluation(question, context)
            batch_results["evaluations"].append(evaluation)

        # Calculate summary statistics
        scores = [eval_result["overall_score"] for eval_result in batch_results["evaluations"]]
        grades = [eval_result["grade"] for eval_result in batch_results["evaluations"]]

        batch_results["summary_statistics"] = {
            "average_score": np.mean(scores),
            "median_score": np.median(scores),
            "std_score": np.std(scores),
            "min_score": np.min(scores),
            "max_score": np.max(scores),
            "grade_distribution": {grade: grades.count(grade) for grade in set(grades)},
            "pass_rate": sum(1 for score in scores if score >= 0.6) / len(scores)
        }

        return batch_results
