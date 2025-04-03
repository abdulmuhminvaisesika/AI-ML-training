# Restaurant Review AI Agent

## Overview
This AI-powered agent answers user queries about a restaurant by analyzing customer reviews. It retrieves relevant reviews using vector embeddings and generates responses using an AI language model.

## Features
- **Natural Language Querying**: Users can ask questions about restaurant experiences.
- **Vector Search for Reviews**: Uses embeddings to find relevant reviews.
- **AI-Powered Answers**: Generates responses based on retrieved reviews.
- **Interactive Chat Interface**: Runs continuously until the user exits.


## Dependencies
Ensure you have the following installed:

- `langchain`
- `langchain_ollama`
- `langchain_chroma`
- `pandas`
- `os`

Install the required dependencies using:
```bash
pip install langchain langchain_ollama langchain_chroma pandas
```

## How It Works
### 1. Review Processing (**vector.py**)
- Loads reviews from `realistic_restaurant_reviews.csv`.
- Converts them into vector embeddings using **OllamaEmbeddings**.
- Stores them in a **Chroma vector database** for quick retrieval.
- Provides a retriever function to fetch relevant reviews.

### 2. Query Handling (**main.py**)
- Loads the AI model (**llama3.2**).
- Takes user input (restaurant-related questions).
- Retrieves the top 5 most relevant reviews from `vector.py`.
- Generates structured AI responses based on the reviews.

## How to Run
### 1. Prepare the dataset
Ensure `realistic_restaurant_reviews.csv` is present in the directory.

### 2. Run the application
```bash
python main.py
```

### 3. Ask Questions
Example:
```
Do people complain about delivery?
```
AI will fetch relevant reviews and generate a response.

### 4. Exit the program
Type `q` and press **Enter**.

