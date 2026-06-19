# 💰 WealthGain — AI-Powered Personal Finance Advisor

> **Agents do the heavy lifting, a human owns the high-stakes calls, and every decision is explainable.**

WealthGain is a production-grade agentic AI system that analyses personal spending data and delivers actionable financial advice. Built as a hands-on AI Architect learning project — every layer from ML training to multi-agent orchestration to observability was built from scratch.

---

## 🏗️ Architecture

```
User (Streamlit Chat UI)
         ↓
LangGraph Supervisor — classifies intent, routes to specialist
         ↓              ↓                ↓
  Triage Agent    Advice Agent      Goal Agent
  (spending data) (RAG + LLM)      (goal timeline)
         ↓              ↓                ↓
         MCP Server (4 tools)
         ↓
   MongoDB (transactions, profiles)
         ↓
   Langfuse (observability)
```

### Agent routing
| Query type | Agent | Tools called |
|---|---|---|
| "Show my spending" | Triage | `get_spending_summary`, `flag_overspending` |
| "Where to cut expenses?" | Advice | `get_spending_summary` + RAG rules |
| "How long to reach my goal?" | Goal | `get_savings_rate`, `project_goal_timeline` |

---

## 🧠 What's inside

### ML layer (trained from scratch)
- **Transaction classifier** — Random Forest, 97.5% accuracy, oversampling to fix class imbalance
- **Anomaly detector** — Isolation Forest, 2% contamination threshold
- **Expense predictor** — Linear Regression, per-user monthly spending forecast

### RAG layer
- 45 personal finance rules embedded with `all-MiniLM-L6-v2`
- Cosine similarity retrieval — top-3 rules per query
- LLM-as-judge evaluation: 4.2/5 precision, 3.6/5 recall

### Agent layer (LangGraph)
- `StateGraph` with 4 nodes: supervisor, triage, advice, goal
- Conditional routing via `add_conditional_edges`
- Shared `WealthGainState` flows through all nodes
- MCP server exposes 4 tools over stdio transport

### MCP server (4 tools)
- `get_spending_summary` — category breakdown + savings rate
- `get_savings_rate` — actual vs recommended (20%) savings
- `flag_overspending` — threshold-based category alerts + anomaly detection
- `project_goal_timeline` — months to reach savings goal

### Observability
- Langfuse traces every node — input, output, latency
- Token-level tracking on advice node
- Prompt caching on finance rules (1337 token static block)

---

## 🗂️ Project structure

```
wealthgain/
  data/
    generate.py           # synthetic transaction generator (2 users, 1316 rows)
    train.py              # ML model training
    transactions.csv      # generated data
    finance_rules.txt     # 45 RAG knowledge base rules
  models/
    classifier.pkl        # trained Random Forest
    anomaly_detector.pkl  # trained Isolation Forest
    expense_predictor.pkl # trained Linear Regression (per user)
  graph/
    state.py              # WealthGainState TypedDict
    nodes.py              # supervisor, triage, advice, goal nodes
    graph.py              # StateGraph definition + conditional edges
  mcp/
    server.py             # FastMCP server — 4 tools
  rag/
    embed_rules.py        # embed finance rules → .npy
    retrieval.py          # cosine similarity retrieval
  eval/
    ragas_eval.py         # LLM-as-judge RAG evaluation
  wealthgain_chat.py      # Streamlit chat UI
  Dockerfile              # containerised deployment
  requirements.txt
```

---

## 🚀 Quick start

### Local

```bash
git clone https://github.com/senthil1410/wealthgain
cd wealthgain
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

pip install -r requirements.txt

# Generate data and train models
python data/generate.py
python data/train.py
python rag/embed_rules.py

# Run
streamlit run wealthgain_chat.py
```

### Docker

```bash
docker build -t wealthgain .
docker run -p 8501:8501 \
  -e ANTHROPIC_API_KEY=your_key \
  -e LANGFUSE_PUBLIC_KEY=your_key \
  -e LANGFUSE_SECRET_KEY=your_key \
  -e LANGFUSE_BASE_URL=https://us.cloud.langfuse.com \
  wealthgain
```

---

## 🔑 Environment variables

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key (claude-sonnet-4-6) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse observability public key |
| `LANGFUSE_SECRET_KEY` | Langfuse observability secret key |
| `LANGFUSE_BASE_URL` | Langfuse host URL |

---

## 📊 ADRs — Key architecture decisions

### ADR-001 — LangGraph over manual agent loop
**Decision:** Use LangGraph StateGraph instead of a hand-written supervisor loop.
**Rationale:** LangGraph provides conditional routing, shared state management, and async node execution out of the box. The manual loop (as built in witt-support) works but requires explicit message history management across agents. LangGraph handles this via the `add_messages` annotation.

### ADR-002 — MCP server for tool exposure
**Decision:** Expose all data operations as MCP tools rather than direct function calls inside nodes.
**Rationale:** MCP is a standard protocol — the same server works with Claude Desktop, MCP Inspector, and any future client without modification. Tool definitions are self-documenting. The transport (stdio locally, HTTPS in production via AgentCore Gateway) is swappable without changing tool logic.

### ADR-003 — Train ML models vs use LLM for classification
**Decision:** Train a Random Forest classifier for transaction categorisation rather than asking the LLM to classify each transaction.
**Rationale:** 1316 transactions classified per session. LLM classification would cost ~$0.02 per session just for categorisation. Random Forest runs in milliseconds at zero marginal cost. LLM reserved for reasoning and advice generation where it adds genuine value.

### ADR-004 — Cosine similarity RAG vs Atlas Vector Search
**Decision:** Pure-Python cosine similarity for RAG retrieval.
**Rationale:** 45 rules is well within brute-force range. No cloud dependency, no additional cost, full control of retrieval logic. Production upgrade path: swap `retrieval.py` for Atlas `$vectorSearch` aggregation — embedding schema stays identical.

### ADR-005 — Prompt caching on finance rules
**Decision:** Cache the static finance rules block (1337 tokens) in the advice node system prompt.
**Rationale:** Finance rules are identical across all advice queries. Caching reduces input token cost by 90% for the rules block on cache hits. Dynamic content (spending summary, flags, user question) injected in the user message — never cached.

---

## 🗺️ Production mapping

| Local | Production (AWS) |
|---|---|
| `streamlit run` | ECS Fargate container |
| CSV file | MongoDB Atlas |
| stdio MCP transport | AgentCore Gateway (HTTPS) |
| `agent_memory.json` | DynamoDB |
| Langfuse cloud | CloudWatch + Langfuse |
| Docker local | ECR → ECS |

---

## 🧪 Evaluation

Run the LLM-as-judge RAG evaluation:

```bash
python eval/ragas_eval.py
```

Results on 5 golden queries:
- Average Precision: **4.2/5**
- Average Recall: **3.6/5**
- Gap: savings rate query misses 50/30/20 rule due to vocabulary mismatch — fix via HyDE

---

## 🔗 Related projects

- **[witt-support](https://github.com/senthil1410/witt-support)** — Healthcare fund management agent: LangGraph pipeline, MCP server (HITL), multi-agent system, AgentCore simulation
- **witt-fraud-detection** — Healthcare fraud detection: FAISS RAG, SHAP explainability, OCR, model drift detection
- **AI Test Agent** — AST-based code scanner + LLM orchestrator: strategy generation, test case gen, regression selection

---

## 👤 Author

**Senthilmurugan K** — Engineering Lead, AI & QE  
20 years across mechanical engineering, IT automation, full-stack, and AI engineering.  
GitHub: [@senthil1410](https://github.com/senthil1410)
