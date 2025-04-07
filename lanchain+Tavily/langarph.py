import os
os.environ["TAVILY_API_KEY"] = "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"

from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.chat_models import ChatOllama

# Define the state
class State(TypedDict):
    input: str
    llm_response: str
    search_result: str

# Initialize Ollama
llm = ChatOllama(model="llama3")

# Tool for web search
search_tool = TavilySearchResults(max_results=1)

# Node: LLM
def llm_node(state: State) -> State:
    message = HumanMessage(content=state["input"])
    result = llm.invoke([message])
    return {"llm_response": result.content}

# Logic to check if LLM is unsure
def should_search(state: State) -> str:
    if "i don't know" in state["llm_response"].lower() or "not sure" in state["llm_response"].lower():
        return "search"
    return "final"

# Node: Web Search
def web_search_node(state: State) -> State:
    results = search_tool.invoke({"query": state["input"]})
    return {"search_result": results[0]["content"]}

# Node: Final Response
def final_node(state: State) -> State:
    if state.get("search_result"):
        return {
            "llm_response": f"{state['llm_response']}\n\n(Added from search: {state['search_result']})"
        }
    return {"llm_response": state["llm_response"]}

# Create graph
builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_node("search", web_search_node)
builder.add_node("final", final_node)

builder.set_entry_point("llm")
builder.add_conditional_edges("llm", should_search)
builder.add_edge("search", "final")
builder.set_finish_point("final")

# ✅ Show Graph
print("🧩 LangGraph Structure:\n")
print(builder.draw_ascii())  # <-- Works in v0.3.25

# Compile the app
app = builder.compile()

# Entry point to run the graph
if __name__ == "__main__":
    response = app.invoke({"input": "What is LangGraph?"})
    print("\n🧠 Response:\n", response["llm_response"])
