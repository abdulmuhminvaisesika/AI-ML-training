# Text Preprocessing
* for referance https://www.geeksforgeeks.org/text-preprocessing-for-nlp-tasks/


* Text preprocessing in AI/ML refers to a series of techniques used to clean and transform raw text data into a format suitable for Natural Language Processing (NLP) tasks, 
* essentially preparing the text by removing unnecessary elements and standardizing it to ensure better performance of machine learning models when analyzing it;
* One of the foundational steps in NLP is text preprocessing, which involves cleaning and preparing raw text data for further analysis or model training. 
*  Proper text preprocessing can significantly impact the performance and accuracy of NLP models. 
  
* Improving Data Quality
* Enhancing Model Performance
* Reducing Complexity


## Text Preprocessing Technique in NLP

### Regular Expressions

* Regular expressions (regex) are a powerful tool in text preprocessing for Natural Language Processing (NLP). 
* They allow for efficient and flexible pattern matching and text manipulation.
  
### Tokenization

* Tokenization is the process of breaking down text into smaller units, such as words or sentences. 
* This is a crucial step in NLP as it transforms raw text into a structured format that can be further analyzed.
  

### stop words
* A stop word is a commonly used word (such as “the”, “a”, “an”, or “in”) that a search engine has been programmed to ignore, both when indexing entries for searching and when retrieving them as the result of a search query. 
* NLTK(Natural Language Toolkit) in Python has a list of stopwords stored in 16 different languages.
* Types of Stopwords:
    * Common Stopwords
    * Custom Stopwords
    * Numerical Stopwords
    * Single-Character Stopwords


### Punctuation


* One essential step in preprocessing text data for NLP tasks is removing punctuations. 
* Punctuation removal simplifies text data, streamlining the analysis by reducing the complexity and variability within the data.
* Removing punctuation also contributes to data uniformity, ensuring that the text is processed in a consistent manner, which is paramount for algorithms to perform optimally. 


### Lemmatization

* Lemmatization is the process of reducing words to their base or dictionary form, known as the lemma.


### Stemming

* Stemming is a more straightforward process that cuts off prefixes and suffixes (i.e., affixes) to reduce a word to its root form. 


### POS(Parts-Of-Speech)

* One of the core tasks in Natural Language Processing (NLP) is Parts of Speech (PoS) tagging, which is giving each word in a text a grammatical category, such as nouns, verbs, adjectives, and adverbs.

* Parts of Speech tagging is a linguistic activity in Natural Language Processing (NLP) wherein each word in a document is given a particular part of speech (adverb, adjective, verb, etc.) or grammatical category.




### n-grams: unigrams, bigrams , trigrams


* N-grams are continuous sequences of N words in a given text. 
* They are useful in natural language processing (NLP) for text analysis, machine learning, and search engines.

1. Unigram (1-gram) → Single word
2. Bigram (2-gram) → Two consecutive words
3. Trigram (3-gram) → Three consecutive words
   

### TF-IDF (Term Frequency-Inverse Document Frequency) Explained

TF-IDF is a numerical statistic used in **Natural Language Processing (NLP)** to measure the importance of words in a document relative to a collection of documents (corpus).  
It helps filter out common words (e.g., **"the"**, **"is"**) and highlights important words.

---

### **Formula for TF-IDF**

#### **1. Term Frequency (TF)**
Measures how frequently a term appears in a document.

**TF = (Number of times a word appears in a document) / (Total words in the document)**  

#### **2. Inverse Document Frequency (IDF)**  
Measures how unique a term is across documents.

**IDF = log(Total number of documents / Number of documents containing the word)**  

#### **3. TF-IDF Score**
TF-IDF is the product of **TF** and **IDF**:

**TF-IDF = TF × IDF**



# Word Embeddings

Word embeddings convert words into continuous dense numerical vectors.  
They capture semantic meaning (words with similar meanings have similar vectors).  
Unlike Bag of Words (BoW), embeddings consider context and word relationships.  

---

## Bag of Words (BoW)

- Represents text as a matrix of word counts or occurrences.  
- Ignores grammar and word order but captures word frequency.  
- Creates a sparse matrix (lots of zeros).  

### **How BoW Works**  
1. **Tokenization**: Convert text into words.  
2. **Build a Vocabulary**: List all unique words in the dataset.  
3. **Create a Vector Representation**: Count how often each word appears.  

---

## Word2Vec

**Word2Vec** is a technique developed by Google to represent words as dense vectors, capturing their meaning, context, and relationships.  

- Converts words into numerical vectors.  
- Helps in NLP tasks like sentiment analysis, chatbots, and machine translation.  
- Unlike Bag of Words (BoW), it preserves semantic meaning.  

### **How Word2Vec Works**  
Word2Vec uses **Neural Networks** to generate word embeddings. It has two architectures:  

### **A) CBOW (Continuous Bag of Words)**  
- Predicts a word based on its surrounding words.  
- Faster and works well with small datasets.  
- **Example**: Given the words **"The ___ is barking"**, predict **"dog"**.  

### **B) Skip-gram Model**  
- Predicts surrounding words given a target word.  
- Works better with large datasets.  
- **Example**: Given the word **"dog"**, predict words like **"barking", "pet", "animal"**.  



---
---

