# nodes/use_tool_node.py

from classes import State
from tools.tavliy_tool import tavily_tool

def use_tool_node(state: State) -> State:
    """
    A node that uses the tavily_tool wrapper to fetch real-time financial data.
    """

    # Get the user’s query from state
    query = state["query"]

    # Call the external tool
    try:
        result = tavily_tool(query)
    except Exception as e:
        result = f"Error fetching data: {str(e)}"

    # Store the result in state
    state["data"] = {"tool_response": result}
    state["message"] = result
    state["next_node"] = "final_response_node"
    return state
