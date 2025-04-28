
from classes import State

def greeting_node(state: State) -> State:
    """
    A node that greets the user and introduces Tydo.
    """
    response = state["model"].invoke(
        """You are Tydo, a friendly and autonomous real-time financial assistant.

        Welcome the user, introduce yourself as Tydo, and briefly mention that you can help with:
        - Real-time financial data
        - Investment insights
        - Budgeting help
        - General financial planning support

        Politely ask the user what they'd like help with today.
        
        """
    )

    
    state["message"] = response
    return state
