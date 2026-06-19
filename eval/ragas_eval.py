import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic
from dotenv import load_dotenv
from rag.retrieval import retrieve_rules

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────
# Golden dataset
# ─────────────────────────────────────────
golden = [
    {
        "question":     "I spend too much on food delivery apps every month",
        "ground_truth": "Food delivery apps are a major budget leak. Cook at home more often.",
    },
    {
        "question":     "How much of my salary should I save each month",
        "ground_truth": "Save at least 20% of income using the 50/30/20 rule.",
    },
    {
        "question":     "My rent takes up a large portion of my income",
        "ground_truth": "Rent should not exceed 30% of take-home income.",
    },
    {
        "question":     "Should I invest in mutual funds or keep money in savings",
        "ground_truth": "SIP in index funds outperforms savings accounts over 10+ years.",
    },
    {
        "question":     "I have credit card debt and want to start investing",
        "ground_truth": "Pay off high-interest debt before investing.",
    },
]

# ─────────────────────────────────────────
# LLM-as-judge scorer
# ─────────────────────────────────────────
def score_retrieval(question: str, retrieved: list[dict], ground_truth: str) -> dict:
    context = "\n".join([f"- {r['rule']}" for r in retrieved])

    prompt = f"""You are evaluating a RAG retrieval system for financial advice.

Question: {question}
Ground truth answer: {ground_truth}

Retrieved context:
{context}

Score the retrieval on two dimensions (1-5 each):
1. Context Precision: Are the retrieved rules relevant to the question?
2. Context Recall: Do the retrieved rules contain the information needed to answer?

Reply in this exact format:
PRECISION: <score>
RECALL: <score>
REASON: <one sentence>"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}]
    )

    text   = response.content[0].text
    lines  = text.strip().split("\n")
    result = {"precision": 0, "recall": 0, "reason": ""}

    for line in lines:
        if line.startswith("PRECISION:"):
            result["precision"] = int(line.split(":")[1].strip())
        elif line.startswith("RECALL:"):
            result["recall"] = int(line.split(":")[1].strip())
        elif line.startswith("REASON:"):
            result["reason"] = line.split(":", 1)[1].strip()

    return result

# ─────────────────────────────────────────
# Run evaluation
# ─────────────────────────────────────────
print("=" * 60)
print("WealthGain RAG Evaluation — LLM-as-Judge")
print("=" * 60)

total_precision = 0
total_recall    = 0

for item in golden:
    retrieved = retrieve_rules(item["question"], top_k=3)
    scores    = score_retrieval(item["question"], retrieved, item["ground_truth"])

    total_precision += scores["precision"]
    total_recall    += scores["recall"]

    print(f"\nQ: {item['question']}")
    print(f"  Top match:  {retrieved[0]['rule'][:60]}... [{retrieved[0]['score']}]")
    print(f"  Precision:  {scores['precision']}/5")
    print(f"  Recall:     {scores['recall']}/5")
    print(f"  Reason:     {scores['reason']}")

n = len(golden)
print(f"\n{'=' * 60}")
print(f"Average Precision: {total_precision/n:.1f}/5")
print(f"Average Recall:    {total_recall/n:.1f}/5")
print(f"Overall RAG Score: {(total_precision + total_recall)/(n*2)*5:.1f}/5")
print(f"{'=' * 60}")