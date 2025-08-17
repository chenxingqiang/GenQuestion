
from datasets import Dataset

# ==================== Data Processor ====================
class QuestionsDataPreparer:
    """Competition problem dataset processor"""

    def __init__(self):
        print("🔄 Initializing data processor...")
        self.real_processor = None
        try:
            from tests.real_math_data_processor import RealMathDataProcessor
            self.real_processor = RealMathDataProcessor()
            print("✅ Real dataset processor loaded")
        except ImportError:
            print("⚠️ Real dataset processor not found, using simulated data")

    def prepare_training_data(self, tokenizer) -> Dataset:
        """Prepare training dataset - Using unsloth standard approach exactly like reference script"""
        print("🔄 Preparing training dataset...")

        # Try to use real data first (following unsloth reference script exactly)
        try:
            if self.real_processor:
                print("✅ Using real math reasoning dataset (unsloth style)")
                return self.real_processor.prepare_unsloth_training_dataset(
                    tokenizer=tokenizer,
                    chat_percentage=0.25  # Same ratio as reference script
                )
        except Exception as e:
            print(f"⚠️ Real data processor failed: {e}")

        # Fallback: Create simple unsloth-style dataset from basic prompts
        print("🔄 Creating fallback unsloth-style training dataset...")

        # Basic problem generation examples in unsloth format
        basic_conversations = [
            [
                {"role": "user", "content": "Generate a challenging algebra competition problem:"},
                {"role": "assistant", "content": "Find all real solutions to the equation x^4 - 5x^2 + 6 = 0.\n\nSolution: Let y = x^2. Then y^2 - 5y + 6 = 0, so (y-2)(y-3) = 0. Thus y = 2 or y = 3, giving x = ±√2 or x = ±√3."}
            ],
            [
                {"role": "user", "content": "Create a geometry problem suitable for math olympiad:"},
                {"role": "assistant", "content": "In triangle ABC, prove that the sum of any two sides is greater than the third side.\n\nSolution: This is the triangle inequality theorem. For any triangle with sides a, b, c: a + b > c, b + c > a, and a + c > b."}
            ],
            [
                {"role": "user", "content": "Design a calculus problem with moderate difficulty:"},
                {"role": "assistant", "content": "Find the maximum value of f(x) = x^3 - 3x^2 + 2 on the interval [0, 3].\n\nSolution: f'(x) = 3x^2 - 6x = 3x(x-2). Critical points at x = 0, 2. Check: f(0) = 2, f(2) = -2, f(3) = 2. Maximum is 2."}
            ],
            [
                {"role": "user", "content": "Formulate a number theory competition question:"},
                {"role": "assistant", "content": "Prove that there are infinitely many prime numbers.\n\nSolution: Assume finitely many primes p1, p2, ..., pn. Consider N = p1·p2·...·pn + 1. N is not divisible by any pi, so either N is prime or has a prime factor not in our list."}
            ],
            [
                {"role": "user", "content": "Develop a combinatorics problem for advanced students:"},
                {"role": "assistant", "content": "How many ways can 8 people be arranged in a circle?\n\nSolution: Fix one person's position to break rotational symmetry. Arrange remaining 7 people in (8-1)! = 7! = 5040 ways."}
            ]
        ]

        # Expand the dataset by creating variations
        all_conversations = basic_conversations.copy()

        # Add variation examples
        for _ in range(10):
            all_conversations.extend([
                [
                    {"role": "user", "content": f"Generate a high-quality competition problem:"},
                    {"role": "assistant", "content": f"Find the roots of x^2 - 4x + 3 = 0.\n\nSolution: Using the quadratic formula: x = (4 ± √(16-12))/2 = (4 ± 2)/2. So x = 3 or x = 1."}
                ],
                [
                    {"role": "user", "content": f"Create an innovative math problem:"},
                    {"role": "assistant", "content": f"Prove that √2 is irrational.\n\nSolution: Assume √2 = p/q in lowest terms. Then 2q^2 = p^2, so p^2 is even, thus p is even. Let p = 2k, then 2q^2 = 4k^2, so q^2 = 2k^2, making q even. Contradiction."}
                ]
            ])

        # Return in unsloth standard format
        return Dataset.from_dict({"conversations": all_conversations})
