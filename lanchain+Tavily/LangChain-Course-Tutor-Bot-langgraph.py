import os
from langgraph.graph import StateGraph
from typing import TypedDict
from langchain_core.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import Runnable
from langchain_ollama import ChatOllama  

# Set Tavily API key
os.environ["TAVILY_API_KEY"] = "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"  # Replace with your key

# Define state structure
class State(TypedDict):
    input: str
    llm_response: str
    search_result: str

# LLM setup
llm = ChatOllama(model="llama3")

# Web search setup
search_tool = TavilySearchResults(max_results=1)

# Node 1: LLM
def llm_node(state: State) -> State:
    message = HumanMessage(content=state["input"])
    result = llm.invoke([message])
    return {"llm_response": result.content}

# Condition: Should we search?
def should_search(state: State) -> str:
    if "I don't know" in state["llm_response"] or "not sure" in state["llm_response"].lower():
        return "search"
    return "final"

# Node 2: Web Search
def web_search_node(state: State) -> State:
    results = search_tool.invoke({"query": state["input"]})
    return {"search_result": results[0]["content"]}

# Node 3: Combine final result
def final_node(state: State) -> State:
    if state.get("search_result"):
        return {
            "llm_response": f"{state['llm_response']}\n\n(Added from search: {state['search_result']})"
        }
    return {"llm_response": state["llm_response"]}

# Build the graph
builder = StateGraph(State)
builder.add_node("llm", llm_node)
builder.add_node("search", web_search_node)
builder.add_node("final", final_node)

builder.set_entry_point("llm")
builder.add_conditional_edges("llm", should_search)
builder.add_edge("search", "final")
builder.set_finish_point("final")

# Compile the graph
app = builder.compile()

# ✅ Interactive loop
if __name__ == "__main__":
    print("📚 LangChain Tutor Bot — ask anything about LangChain!")
    print("💡 Type 'q' to quit.\n")

    while True:
        user_input = input("🔤 Ask a LangChain question: ")
        if user_input.lower() == "q":
            print("👋 Goodbye!")
            break

        result = app.invoke({"input": user_input})
        print("\n🧠 Final Answer:\n", result["llm_response"])
        print("\n" + "-"*60 + "\n")
