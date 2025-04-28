#!/usr/bin/env python3

from typing_extensions import TypedDict
from typing import Optional, Dict
from langchain_ollama import OllamaLLM

class State(TypedDict):
    """
    A class to store the state of a graph for Tydo, the financial assistant.
    """
    query: str                     
    message: str                   
    next_node: str                 
    data: Optional[Dict]           
    model: OllamaLLM
                   

if __name__ == "__main__":
    print("Checking imports")  # Simple check to ensure no import errors
