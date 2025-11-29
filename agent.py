import os
import json
from langgraph.graph import StateGraph, END, START
from langchain_core.rate_limiters import InMemoryRateLimiter
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tools import get_rendered_html, download_file, post_request, run_code, add_dependencies
from typing import TypedDict, Annotated, List, Any
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")
RECURSION_LIMIT = 5000

# -------------------------
# STATE
# -------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]


TOOLS = [run_code, get_rendered_html, download_file, post_request, add_dependencies]

# -------------------------
# GEMINI LLM
# -------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=9 / 60,
    check_every_n_seconds=1,
    max_bucket_size=9
)

llm = init_chat_model(
    model_provider="google_genai",
    model="gemini-2.5-flash",
    rate_limiter=rate_limiter
).bind_tools(TOOLS)

# -------------------------
# SYSTEM PROMPT
# -------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Return answers ONLY using the submit URL specified on the page.

Always:
- Load the page
- Extract instructions
- Solve correctly
- Submit via the correct endpoint
- Follow next URL if provided
- Stop ONLY when no new URL is given
- Then output: END

Use:
- email={EMAIL}
- secret={SECRET}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="messages")
])
llm_with_prompt = prompt | llm

# -------------------------
# AGENT NODE
# -------------------------
def agent_node(state: AgentState):
    resp = llm_with_prompt.invoke({"messages": state["messages"]})
    return {"messages": state["messages"] + [resp]}

# -------------------------
# ROUTING LOGIC
# -------------------------
def route(state):
    last = state["messages"][-1]

    # Tool calls?
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        return "tools"

    # Check END signal
    content = getattr(last, "content", None)
    if isinstance(content, str) and content.strip() == "END":
        return END
    if isinstance(content, list):
        if len(content) and content[0].get("text", "").strip() == "END":
            return END

    return "agent"

# -------------------------
# GRAPH BUILDING
# -------------------------
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_conditional_edges("agent", route)

app = graph.compile()


# -------------------------
# RUN AGENT WITH STRUCTURED RETURN
# -------------------------
def run_agent(url: str):
    """
    Runs the LangGraph agent synchronously and extracts a structured result.
    """

    execution = app.invoke(
        {"messages": [{"role": "user", "content": url}]},
        config={"recursion_limit": RECURSION_LIMIT},
    )

    # Retrieve ALL messages from the LangGraph run
    msgs = execution["messages"]

    # Extract history
    history = []
    for m in msgs:
        entry = {}
        if hasattr(m, "content"):
            entry["content"] = m.content
        if hasattr(m, "tool_calls"):
            entry["tool_calls"] = m.tool_calls
        history.append(entry)

    # Extract final answer (if any)
    final_answer = None
    for m in reversed(msgs):
        if hasattr(m, "content") and isinstance(m.content, str):
            try:
                obj = json.loads(m.content)
                if "answer" in obj:
                    final_answer = obj["answer"]
                    break
            except Exception:
                continue

    return {
        "status": "completed",
        "steps": len(history),
        "final_answer": final_answer,
        "history": history
    }
