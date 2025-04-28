from classes import State
from langchain_core.prompts import PromptTemplate

def use_llm_node(state: State) -> State:
    """
    This node uses the LLM to answer the user's query directly,
    without relying on real-time tools.
    """

    prompt = """You are a helpful and knowledgeable financial assistant.
Answer the user's question in a clear and friendly way.

User: {user_input}
Assistant:
"""

    prompt_template = PromptTemplate.from_template(prompt)

    # Generate response using the model
    response = state["model"].invoke(prompt_template.invoke({
        "user_input": state["query"]
    }))

    # Save the message and move to final node
    state["message"] = response
    state["next_node"] = "final_response_node"
    return state
