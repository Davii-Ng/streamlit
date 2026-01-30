from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
import getpass
from dotenv import load_dotenv


class RAG_pipeline():
    def __init__(self):
        pass






def main():
    load_dotenv()
    
    GEMINI_API = os.getenv("GEMINI_API_KEY")

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
    ai_msg = model.invoke(messages)
    print(ai_msg.text)


if __name__ == '__main__':
    main()