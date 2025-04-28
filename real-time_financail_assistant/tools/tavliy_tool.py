import os
from langchain_community.tools.tavily_search import TavilySearchResults

# Set the environment variable
os.environ["TAVILY_API_KEY"] = "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"

def tavily_tool(query: str):  # Accept query as an argument
    return TavilySearchResults(query=query, k=3)
