import pandas as pd
import os
import re
import nltk
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# Download stopwords & wordnet if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords & lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load restaurant reviews dataset
df = pd.read_csv(r"C:\Users\abdul.muhmin\Downloads\realistic_restaurant_reviews.csv")




# Convert each row into a structured Document format
documents = []
for _, row in df.iterrows():
    doc = Document(
        page_content=row["Review"],  # Use the review text as the content
        metadata={
            "title": row["Title"],
            "date": row["Date"],
            "rating": row["Rating"],
        }
    )
    documents.append(doc)


# Initialize embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Initialize Chroma vector store
vectorstore = Chroma.from_documents(documents, embedding=embeddings, persist_directory="./chroma_db")


def extract_keywords(query):
    """Extracts meaningful keywords from the user's query using lemmatization."""
    # Tokenize and filter words
    words = [
        lemmatizer.lemmatize(word) 
        for word in re.findall(r'\b\w+\b', query.lower())  # Tokenize to words
        if word not in stop_words and len(word) > 1
    ]
    keywords = set(words)  # Remove duplicates by converting to set
    print(keywords)
    print("\n\n-------------------------------------------")
    return keywords

def retrieve_reviews(query):
    """Retrieves the most relevant reviews using ChromaDB."""
    keywords = extract_keywords(query)

    # If similarity_search expects a space-separated string, your current approach is fine.
    query_string = " ".join(keywords)

    # Retrieve top 5 relevant reviews
    results = vectorstore.similarity_search(query_string, k=5)  
    
    if results:  # This is a bit more Pythonic.
        reviews_text = "\n\n".join([f"Review: {res.page_content}" for res in results])
        print(reviews_text)
        print("\n\n---------------------------------------------------------------------------------------------------------------")
        return reviews_text
    else:
        return "No relevant reviews found."
