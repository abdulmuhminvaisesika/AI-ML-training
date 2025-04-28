import os
import sys
sys.path.insert(0, './scripts/')

from langgraph.graph import StateGraph, START, END
from classes import State
from nodes.final_response_node import final_response_node
from nodes.greeting_node import greeting_node
from nodes.tool_decision_node import tool_decision_node
from nodes.use_llm_node import use_llm_node
from nodes.use_tool_node import use_tool_node

from langchain_ollama import OllamaLLM

def main():
    # Initialize the model
    model = OllamaLLM(model="mistral:latest")

    # Create a StateGraph for the workflow
    workflow = StateGraph(State)

    # Add nodes to the workflow
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("query_handling", tool_decision_node)
    workflow.add_node("use_llm", use_llm_node)
    workflow.add_node("use_tool", use_tool_node)
    workflow.add_node("final_response", final_response_node)

    # Add edges to connect the nodes in the workflow
    workflow.add_edge(START, "greeting")
    workflow.add_edge("greeting", "query_handling")
    workflow.add_conditional_edges("query_handling", lambda state: state["next_node"],
                                   ["use_llm", "use_tool"])
    workflow.add_edge("use_llm", "final_response")
    workflow.add_edge("use_tool", "final_response")
    workflow.add_edge("final_response", END)

    # Compile the graph
    graph = workflow.compile()


    # Initialize the state with all necessary keys
    state = {
        "query": "",
        "message": "",
        "next_node": "greeting",
        "data": {},  # Initialize the data field (this will avoid KeyError)
        "model": model 
    }
    # Loop for handling user input and invoking the workflow
    while True:
        
        response = graph.invoke(state)  # Pass the full state to the graph
        print(response["message"])

        break
if __name__ == "__main__":

    main()
