import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langgraph.graph import StateGraph, END
from state import WealthGainState
from nodes import supervisor_node, triage_node, advice_node, goal_node

def route_intent(state: dict) -> str:
    intent = state.get("intent", "TRIAGE")
    if intent == "ADVICE":
        return "advice"
    elif intent == "GOAL":
        return "goal"
    return "triage"

def build_graph():
    g = StateGraph(WealthGainState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("triage",     triage_node)
    g.add_node("advice",     advice_node)
    g.add_node("goal",       goal_node)

    g.set_entry_point("supervisor")

    g.add_conditional_edges(
        "supervisor",
        route_intent,
        {
            "triage": "triage",
            "advice": "advice",
            "goal":   "goal",
        }
    )

    g.add_edge("triage", END)
    g.add_edge("advice", END)
    g.add_edge("goal",   END)

    return g.compile()

async def run(query: str, user_id: str = "USR001"):
    graph = build_graph()
    from langchain_core.messages import HumanMessage

    result = await graph.ainvoke({
        "messages": [HumanMessage(content=query)],
        "user_id":  user_id,
    })
    return result["final_answer"]

if __name__ == "__main__":
    queries = [
        ("Show me my spending breakdown", "USR001"),
        ("Where should I cut my expenses?", "USR001"),
        ("How long will it take to save ₹2 lakhs?", "USR002"),
    ]

    for query, uid in queries:
        print(f"\n>>> {query} [{uid}]")
        answer = asyncio.run(run(query, uid))
        print(f">>> {answer[:200]}...")
        print("=" * 60)