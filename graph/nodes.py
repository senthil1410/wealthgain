import os
import json
import sys
import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()
from langfuse import observe
from langfuse import Langfuse

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL"),
)
client     = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MCP_SERVER = os.path.join(BASE_DIR, "mcp", "server.py")

def mcp_tool_to_anthropic(tool):
    return {
        "name":         tool.name,
        "description":  tool.description,
        "input_schema": tool.inputSchema,
    }
@observe()
def supervisor_node(state: dict) -> dict:
    last_message = state["messages"][-1].content
    user_id      = state.get("user_id", "USR001")

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=20,
        system="""Classify the user query into exactly one:
TRIAGE   - spending summary, breakdown, categories, anomalies
ADVICE   - recommendations, tips, how to save, what to cut
GOAL     - goal tracking, timeline, savings projection
Reply with only one word: TRIAGE, ADVICE, or GOAL""",
        messages=[{"role": "user", "content": last_message}],
    )

    intent = response.content[0].text.strip().upper()
    if "ADVICE" in intent:
        intent = "ADVICE"
    elif "GOAL" in intent:
        intent = "GOAL"
    else:
        intent = "TRIAGE"

    return {"intent": intent, "user_id": user_id}
@observe()
async def triage_node(state: dict) -> dict:
    user_id      = state.get("user_id", "USR001")
    last_message = state["messages"][-1].content

    server_params = StdioServerParameters(
        command=sys.executable, args=[MCP_SERVER])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools           = await session.list_tools()
            anthropic_tools = [mcp_tool_to_anthropic(t) for t in tools.tools]
            messages        = [{"role": "user", "content": f"User {user_id}: {last_message}"}]
            system          = f"You are a financial triage specialist. Use tools to get spending data for user {user_id}."

            while True:
                response = client.messages.create(
                    model="claude-sonnet-4-6",
                    max_tokens=1000,
                    system=system,
                    tools=anthropic_tools,
                    messages=messages,
                )

                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            result = await session.call_tool(block.name, arguments=block.input)
                            tool_results.append({
                                "type":        "tool_result",
                                "tool_use_id": block.id,
                                "content":     result.content[0].text,
                            })
                    messages.append({"role": "user", "content": tool_results})
                else:
                    answer = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            answer += block.text
                    from langchain_core.messages import AIMessage
                    return {
                        "spending_data": {"summary": answer},
                        "final_answer":  answer,
                        "messages":      [AIMessage(content=answer)],
                    }
@observe()
async def advice_node(state: dict) -> dict:
    sys.path.insert(0, BASE_DIR)
    from rag.retrieval import retrieve_rules

    user_id      = state.get("user_id", "USR001")
    last_message = state["messages"][-1].content

    server_params = StdioServerParameters(
        command=sys.executable, args=[MCP_SERVER])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result  = await session.call_tool("get_spending_summary", {"user_id": user_id})
            summary = result.content[0].text
            result2 = await session.call_tool("flag_overspending", {"user_id": user_id})
            flags   = result2.content[0].text

    rules    = retrieve_rules(last_message, top_k=3)
    rule_ctx = "\n".join([f"- {r['rule']}" for r in rules])
    all_rules = open(os.path.join(BASE_DIR, "data", "finance_rules.txt")).read()

    system_text = f"""You are WealthGain, a personal financial advisor for Indian households.
Give specific, actionable, numbered advice based on real spending data.
Always reference actual rupee amounts. Prioritise highest-impact changes first.
Never give generic advice — ground everything in the user's actual numbers.

Complete financial knowledge base:
{all_rules}

Guidelines:
- Flag spending above category thresholds immediately
- Calculate exact rupee amount the user needs to cut
- Suggest specific alternatives with savings estimates
- End with a prioritised 3-step action plan
- Be direct and honest even if numbers are alarming"""

    response = client.beta.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        betas=["prompt-caching-2024-07-31"],
        system=[
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        messages=[
            {
                "role": "user",
                "content": f"""Here is my spending data:

Spending summary:
{summary}

Overspending flags:
{flags}

Relevant financial rules for my question:
{rule_ctx}

My question: {last_message}"""
            }
        ],
    )

    answer = ""
    for block in response.content:
        if hasattr(block, "text"):
            answer += block.text

    from langchain_core.messages import AIMessage
    return {
        "retrieved_rules": rules,
        "final_answer":    answer,
        "messages":        [AIMessage(content=answer)],
    }
@observe()
async def goal_node(state: dict) -> dict:
    user_id      = state.get("user_id", "USR001")
    last_message = state["messages"][-1].content

    server_params = StdioServerParameters(
        command=sys.executable, args=[MCP_SERVER])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result          = await session.call_tool("get_savings_rate", {"user_id": user_id})
            savings_data    = json.loads(result.content[0].text)
            monthly_savings = savings_data.get("total_saved", 0) / 6
            result2         = await session.call_tool(
                "project_goal_timeline",
                {
                    "user_id":         user_id,
                    "goal_amount":     200000,
                    "monthly_savings": max(monthly_savings, 1000),
                }
            )
            timeline = result2.content[0].text

    system = f"""You are WealthGain goal tracker.
Savings data: {json.dumps(savings_data)}
Goal timeline: {timeline}
Give a clear progress update and actionable next steps."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=800,
        system=system,
        messages=[{"role": "user", "content": last_message}],
    )

    # answer = response.content[0].text
    # from langchain_core.messages import AIMessage
    # return {
    #     "final_answer": answer,
    #     "messages":     [AIMessage(content=answer)],
    # }
    try:
        answer = ""
        for block in response.content:
            if hasattr(block, "text"):
                answer += block.text
    except Exception as e:
        answer = f"Error generating advice: {str(e)}"

    from langchain_core.messages import AIMessage
    return {
        "retrieved_rules": rules,
        "final_answer":    answer,
        "messages":        [AIMessage(content=answer)],
    }