import os
import sys
import asyncio
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.graph import build_graph
from langchain_core.messages import HumanMessage

# ─────────────────────────────────────────
# Page config
# ─────────────────────────────────────────
st.set_page_config(
    page_title="WealthGain",
    page_icon="💰",
    layout="wide",
)

# ─────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────
if "current_user" not in st.session_state:
    st.session_state.current_user = "USR001"
if "messages"     not in st.session_state:
    st.session_state.messages     = []
if "display"      not in st.session_state:
    st.session_state.display      = []
if "pending"      not in st.session_state:
    st.session_state.pending      = None

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.title("💰 WealthGain")
    st.caption("Your AI Financial Advisor")
    st.divider()

    user_id = st.selectbox(
        "Select user",
        ["USR001", "USR002"],
        format_func=lambda x: "Arjun Sharma" if x == "USR001" else "Priya Nair",
    )

    # Clear chat when user switches
    if st.session_state.current_user != user_id:
        st.session_state.current_user = user_id
        st.session_state.messages     = []
        st.session_state.display      = []
        st.session_state.pending      = None

    st.divider()
    st.subheader("Quick actions")

    if st.button("📊 Spending breakdown"):
        st.session_state.pending = "Show me my spending breakdown"
    if st.button("✂️ Where to cut expenses"):
        st.session_state.pending = "Where should I cut my expenses?"
    if st.button("🎯 Goal progress"):
        st.session_state.pending = "How long will it take to reach my savings goal?"
    if st.button("⚠️ Unusual transactions"):
        st.session_state.pending = "Are there any unusual transactions I should know about?"

    st.divider()
    st.caption("Powered by LangGraph + MCP + RAG")

# ─────────────────────────────────────────
# Main area
# ─────────────────────────────────────────
name = "Arjun Sharma" if user_id == "USR001" else "Priya Nair"
st.title(f"💰 WealthGain — {name}")
st.caption("Ask anything about your spending, savings, and financial goals.")

# ─────────────────────────────────────────
# Agent runner
# ─────────────────────────────────────────
async def run_agent(query: str, uid: str) -> tuple[str, str]:
    graph  = build_graph()
    result = await graph.ainvoke({
        "messages": [HumanMessage(content=query)],
        "user_id":  uid,
    })
    return result.get("final_answer", ""), result.get("intent", "unknown")

def get_answer(query: str, uid: str) -> tuple[str, str]:
    return asyncio.run(run_agent(query, uid))

def stream_response(text: str):
    import time
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.01)

# ─────────────────────────────────────────
# Render existing chat history
# ─────────────────────────────────────────
for entry in st.session_state.display:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("agent"):
            st.caption(f"Routed to: {entry['agent']} agent")

# ─────────────────────────────────────────
# Process pending query (from button OR input)
# set by button clicks above OR chat input below
# ─────────────────────────────────────────
if prompt := st.chat_input("Ask WealthGain..."):
    st.session_state.pending = prompt

if st.session_state.pending:
    query = st.session_state.pending
    st.session_state.pending = None  # clear immediately

    # Show user message
    st.session_state.display.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run agent and stream response
    with st.chat_message("assistant"):
        with st.spinner("Analysing your finances..."):
            answer, intent = get_answer(query, user_id)
        st.write_stream(stream_response(answer))
        st.caption(f"Routed to: {intent} agent")

    # Save to history
    st.session_state.display.append({
        "role":    "assistant",
        "content": answer,
        "agent":   intent,
    })
    st.session_state.messages.append({"role": "user",      "content": query})
    st.session_state.messages.append({"role": "assistant", "content": answer})