from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
import getpass
from dotenv import load_dotenv
from langchain_chroma import Chroma
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances


class RAG_pipeline():
    def __init__(self):
        pass






def main():
    load_dotenv()
    
    GEMINI_API = os.getenv("GEMINI_API_KEY")
    CHROMA_API = os.getenv("CHROMA_API_KEY")

    model = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        temperature=1.0,  
        max_tokens=None,
        timeout=None,
        max_retries=2,
        api_key = GEMINI_API
    )

    messages = [
        (
            "system",
            "You are a helpful assistant that translates English to French. Translate the user sentence.",
        ),
        ("human", "I love programming."),
    ]

    
    # ai_msg = model.invoke(messages)
    # print(ai_msg.text)


    query_embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001", task_type = "RETRIEVAL_QUERY")
    doc_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT")

    q_embed = query_embeddings.embed_query("What is the capital of France?")
    d_embed = doc_embeddings.embed_documents(
    ["The capital of France is Paris.", "Philipp is likes to eat pizza."]
)

    for i, d in enumerate(d_embed):
        print(f"Document {i + 1}:")
        print(f"Cosine similarity with query: {cosine_similarity([q_embed], [d])[0][0]}")
        print(f"Cosine distance with query: {cosine_distances([q_embed], [d])[0][0]}")
        print(f"Manhanttan distance with query: {manhattan_distances([q_embed], [d])[0][0]}")
        print(f"Eucledian with query: {euclidean_distances([q_embed], [d])[0][0]}")
        print("---")
    

if __name__ == '__main__':
    main()