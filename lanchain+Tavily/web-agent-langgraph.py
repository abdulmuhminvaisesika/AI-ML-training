import os
os.environ["TAVILY_API_KEY"] = "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"  

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langchain_community.chat_models import ChatOllama
from langchain_community.tools.tavily_search import TavilySearchResults

# ---------- STATE ----------
class State(TypedDict):
    input: str
    llm_response: str
    search_result: str
    summary: str

# ---------- LLM ----------
llm = ChatOllama(model="llama3")

def llm_node(state: State) -> State:
    print("🧠 LLM Node")
    message = HumanMessage(content=state["input"])
    result = llm.invoke([message])
    return {"llm_response": result.content}



# ---------- WEB SEARCH ----------
search_tool = TavilySearchResults(max_results=1)

def search_node(state: State) -> State:
    print("🔍 Search Node")
    results = search_tool.invoke({"query": state["input"]})
    return {"search_result": results[0]["content"]}

# ---------- SUMMARIZATION ----------
def summarization_node(state: State) -> State:
    print("✍️ Summary Node")
    message = HumanMessage(content=f"Summarize this:\n\n{state['search_result']}")
    summary = llm.invoke([message])
    return {"summary": summary.content}

# ---------- FINAL OUTPUT ----------
def final_node(state: State) -> State:
    print("✅ Final Node")
    if state.get("summary"):
        return {
            "llm_response": f"{state['llm_response']}\n\n(Info from search: {state['summary']})"
        }
    return {"llm_response": state["llm_response"]}

# ---------- BUILD GRAPH ----------
builder = StateGraph(State)

builder.add_node("llm", llm_node)
builder.add_node("search", search_node)
builder.add_node("summarize", summarization_node)
builder.add_node("final", final_node)

builder.set_entry_point("llm")
builder.add_edge("llm", "search")
builder.add_edge("search", "summarize")
builder.add_edge("summarize", "final")
builder.set_finish_point("final")

app = builder.compile()

# ---------- RUN ----------
if __name__ == "__main__":
    user_input = input("🔤 Ask me anything: ")
    result = app.invoke({"input": user_input})
    print("\n🧠 Final Answer:\n", result["llm_response"])
