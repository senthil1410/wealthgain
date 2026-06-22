import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualPrecisionMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
import anthropic

# ── Custom Anthropic model wrapper for DeepEval ───────────────────────────────
class AnthropicModel(DeepEvalBaseLLM):
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model  = "claude-sonnet-4-6"

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return "claude-sonnet-4-6"

# ── Sample test cases (query + retrieved context + actual response) ───────────
model = AnthropicModel()

test_cases = [
    LLMTestCase(
        input="I spend a lot on food delivery every month. What should I do?",
        actual_output=(
            "You should reduce food delivery spending. "
            "Try cooking at home at least 4 days a week. "
            "Food delivery apps are a major budget leak according to the 50/30/20 rule."
        ),
        expected_output="Reduce food delivery, cook at home more often.",
        retrieval_context=[
            "Food delivery apps are a major budget leak — cook at home at least 4 days a week.",
            "The 50/30/20 rule: allocate 50% of income to needs, 30% to wants, and 20% to savings.",
            "Restaurant and cafe spending above Rs 3000 per month needs review — cook at home more."
        ]
    ),
    LLMTestCase(
        input="How much should I save every month?",
        actual_output=(
            "You should save at least 20% of your income every month. "
            "Automate savings on salary day before spending on anything else. "
            "Even small SIPs started early benefit from compound interest."
        ),
        expected_output="Save at least 20% of income, automate it on salary day.",
        retrieval_context=[
            "The 50/30/20 rule: allocate 50% of income to needs, 30% to wants, and 20% to savings.",
            "Pay yourself first — automate savings before spending on discretionary items.",
            "Compound interest works best when started early — even small SIPs matter."
        ]
    ),
    LLMTestCase(
        input="Should I invest or pay off my credit card first?",
        actual_output=(
            "Pay off your credit card first. "
            "High-interest debt should always be cleared before investing. "
            "Credit card bills must be paid in full every month to avoid interest charges."
        ),
        expected_output="Clear high-interest debt before investing.",
        retrieval_context=[
            "High-interest debt should be paid off before investing.",
            "Credit card bills must be paid in full every month to avoid interest.",
            "The 50/30/20 rule: allocate 50% of income to needs, 30% to wants, and 20% to savings."
        ]
    ),
]

# ── Metrics ───────────────────────────────────────────────────────────────────
metrics = [
    AnswerRelevancyMetric(threshold=0.7, model=model, verbose_mode=True),
    FaithfulnessMetric(threshold=0.7, model=model, verbose_mode=True),
    ContextualPrecisionMetric(threshold=0.7, model=model, verbose_mode=True),
]

# ── Run evaluation ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔍 Running DeepEval metrics on WealthGain RAG responses...\n")
    results = evaluate(test_cases=test_cases, metrics=metrics)
    print("\n✅ Evaluation complete.")