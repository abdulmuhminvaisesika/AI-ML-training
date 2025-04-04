from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from vector import retrieve_reviews
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain

model = OllamaLLM(model="llama3.2")

# Use memory that keeps only the last 5 interactions
memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10)


template = """
        You are a helpful and conversational AI assistant helping users with questions about a pizza restaurant based on customer reviews.

        ### Conversation History:
        {chat_history}

        ### User Input:
        {input}



        ### Instructions:
        - Respond in a friendly, engaging tone.
        - Use relevant details from the reviews to support your answer.
        - Reference past questions if helpful.
        - Keep your answer short, clear, and helpful.

        ### Your Answer:
    """


prompt = PromptTemplate.from_template(template)

chain = LLMChain(llm=model, prompt=prompt, memory=memory)


while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    if question.lower() == "q":
        break

    reviews = retrieve_reviews(question)
    full_input = f"Reviews:\n{reviews}\n\nQuestion:\n{question}"
    result = chain.invoke({"input": full_input})
    print(result["text"])