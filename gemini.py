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
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from transformers import GPT2Tokenizer



class RAG_pipeline():
    def __init__(self):
        pass






def main():
    load_dotenv()
    
    GEMINI_API = os.getenv("GEMINI_API_KEY")
    CHROMA_API = os.getenv("CHROMA_API_KEY")

    # model = ChatGoogleGenerativeAI(
    #     model="gemini-3-flash-preview",
    #     temperature=1.0,  
    #     max_tokens=None,
    #     timeout=None,
    #     max_retries=2,
    #     api_key = GEMINI_API
    # )

    # messages = [
    #     (
    #         "system",
    #         "You are a helpful assistant that translates English to French. Translate the user sentence.",
    #     ),
    #     ("human", "I love programming."),
    # ]

    
    # # ai_msg = model.invoke(messages)
    # # print(ai_msg.text)

    loader = PyPDFLoader(
    "./Multi-RAG.pdf",
    mode="single",
    pages_delimiter="\n-------THIS IS A CUSTOM END OF PAGE-------\n",
)
    docs = loader.load()


    

    text_splitter = RecursiveCharacterTextSplitter( chunk_size=1000, chunk_overlap=200, add_start_index = True)
    texts = text_splitter.split_documents(docs)



    query_embeddings = GoogleGenerativeAIEmbeddings(model = "models/gemini-embedding-001", task_type = "RETRIEVAL_QUERY")
    doc_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT")

    # q_embed = query_embeddings.embed_query("What is Multi-RAG?")
    # d_embed = doc_embeddings.embed_documents(texts)
    
    vector_store = Chroma(
    collection_name="example_collection",
    embedding_function= doc_embeddings,
    persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not necessary
)
    ids = vector_store.add_documents(documents = texts)
    
    embedding = doc_embeddings.embed_query("What is RAG? ")
    results = vector_store.similarity_search_by_vector(embedding)
    print(results)


if __name__ == '__main__':
    main()