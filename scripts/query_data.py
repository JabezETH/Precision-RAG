import os
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables from .env file
load_dotenv()

CHROMA_PATH = "/home/jabez/Documents/week_7/Precision-RAG/chroma"

PROMPT_TEMPLATE = """
You are a chatbot with expertise in Ethiopian criminal law.
Answer the question based only on the following context. Make sure to explicitly mention the article number where you get the information:

{context}

---

Answer the question based on the above context, and include the relevant article numbers explicitly in your response: {question}
"""

def process_message(query_text):
    if not isinstance(query_text, str):
        query_text = str(query_text)  # Convert to string if necessary

    # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print("Unable to find matching results.")
        return

    # Format the context with article numbers
    context_pieces = []
    sources = []
    for doc, _score in results:
        source = doc.metadata.get("source", "Unknown source")
        article = doc.metadata.get("article", "Unknown article")
        context_pieces.append(f"{doc.page_content} (Article: {article})")
        sources.append(article)

    context_text = "\n\n---\n\n".join(context_pieces)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Generate the response
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    # Ensure article numbers are mentioned explicitly
    for article in sources:
        if article in response_text:
            response_text = response_text.replace(article, f"Article {article}")

    formatted_response = f"Response: {response_text}\nSources: {', '.join(sources)}"
    return formatted_response

if __name__ == "__main__":
    query = "What is the procedure for arrest under Ethiopian criminal law?"
    response = process_message(query)
    print(response)
