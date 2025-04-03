from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
    You are a helpful assistant answering questions about a pizza restaurant based on customer reviews.

    ### How to Respond:
    - Give a clear and friendly answer.
    - Use key details from the reviews to support your response.
    - Keep it short, engaging, and to the point.

    ### Reviews:
    {reviews}

    ### Question:
    {question}

    ### Your Answer:
    """

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question == "q":
        break
    
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)