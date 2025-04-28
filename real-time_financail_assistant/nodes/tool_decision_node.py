from classes import State
from langchain_core.prompts import PromptTemplate

def tool_decision_node(state: State) -> State:
    """
    A node that decides if the user input requires a real-time tool (Tavily)
    or can be answered directly by the LLM.
    """

    prompt = """You are an assistant that classifies user input into either 
                "USE_TOOL" or "USE_LLM" or "UNDEFINED". Your response should be strictly one 
                of these three options.

                Use "USE_TOOL" if the query needs real-time or latest information (like live news, stock prices, current events).
                Use "USE_LLM" if the query is general knowledge, advice, or explanation that doesn't need live data.
                Use "UNDEFINED" if the question is unclear or unrelated.

                Examples:
                User: What is the current stock price of Apple?
                AI: USE_TOOL

                User: How does a mutual fund work?
                AI: USE_LLM


                User: {user_input}
                AI:
            """

    prompt_template = PromptTemplate.from_template(prompt)

    for _ in range(3):

        state["query"] = input(">>> ")

        decision = state["model"].invoke(prompt_template.invoke({"user_input": state["query"]})).strip().upper()

        if decision in ["USE_TOOL", "USE_LLM"]:
            state["next_node"] = "use_tool" if decision == "USE_TOOL" else "use_llm"
            return state

        print("Couldn't classify the query. Trying again...")

    print("Sorry, unable to classify the request. Please rephrase and try again.")
    exit()
