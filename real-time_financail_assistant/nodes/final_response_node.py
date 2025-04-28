from classes import State
from langchain_core.prompts import PromptTemplate

def final_response_node(state: State) -> State:
    """
    Uses the LLM to format the raw message into a well-structured, conversational response.
    """

    raw_response = state.get("message", "")

    formatting_prompt = """
You are a friendly and professional financial assistant. Format the following financial information into a clear, user-friendly response.

- Make it conversational but informative.
- Use bullet points or short paragraphs.
- If it includes sensitive or time-based data, add a disclaimer.
- Encourage follow-up questions.
- Never include links.

Raw info:
{raw_response}

Your response:
"""

    prompt = PromptTemplate.from_template(formatting_prompt)

    formatted = state["model"].invoke(prompt.invoke({"raw_response": raw_response})).strip()

    print(f"\n🤖 {formatted}\n")

    state["next_node"] = "end"
    return state
