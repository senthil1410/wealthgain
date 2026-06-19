from typing import Annotated
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class WealthGainState(TypedDict):
    """Shared state flowing through all nodes in the graph."""
    messages:        Annotated[list, add_messages]  # full conversation history
    user_id:         str                            # which user is asking
    intent:          str                            # classified intent
    spending_data:   dict                           # populated by triage node
    retrieved_rules: list                           # populated by RAG
    final_answer:    str                            # populated by advice/goal node