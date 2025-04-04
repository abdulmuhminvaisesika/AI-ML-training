import pandas as pd
import os
import re
from langchain_ollama import OllamaEmbeddings
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords & wordnet if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')


# Initialize stopwords & lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Load restaurant reviews dataset
df = pd.read_csv(r"C:\Users\abdul.muhmin\Downloads\realistic_restaurant_reviews.csv")

# Initialize embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def extract_keywords(query):
    """Extracts meaningful keywords from the user's query using lemmatization."""
    
    # Remove possessive 's (e.g., "what's" → "what")
    query = re.sub(r"(\w+)'s", r"\1", query)  
    
    words = re.findall(r'\b\w+\b', query.lower())  # Tokenization
    filtered_words = [
        lemmatizer.lemmatize(word) for word in words 
        if word not in stop_words and len(word) > 1
    ]
    print(list(set(filtered_words)))
    return list(set(filtered_words))  # Remove duplicates


def retrieve_reviews(query):
    """Searches for reviews that contain extracted keywords."""
    keywords = extract_keywords(query)
    if not keywords:
        return "No relevant keywords found in your question."

    # Filter reviews containing any of the keywords
    matching_reviews = df[df['Review'].str.lower().apply(lambda review: any(word in review for word in keywords))]

    if matching_reviews.empty:
        return "No relevant reviews found."

    # Return top 5 reviews as formatted text
    top_reviews = matching_reviews.head(5)


    reviews_text = "\n\n".join([f"Rating: {row['Rating']}, Review: {row['Review']}" for _, row in top_reviews.iterrows()])
    

    print(reviews_text)
    print("--------------------")

    return reviews_text
