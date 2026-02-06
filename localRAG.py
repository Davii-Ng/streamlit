import os
from dotenv import load_dotenv
from transformers import GPT2TokenizerFast
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM





def load_docs(pdf_path):
    """Load multiple PDF files from one directory"""
    documents = []
    

    """Safety Checks"""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Folder does not exist: {pdf_path}")

    if not os.path.isdir(pdf_path):
        raise NotADirectoryError(f"Path is not a directory: {pdf_path}")
    

    """Load PDFs"""
    for file in os.listdir(pdf_path):
        if file.endswith('.pdf'):
            file_path = os.path.join(pdf_path, file)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

    return documents




def Recursive_Text_Splitter(document, chunk_size, chunk_overlap):
    """Traditional Text Splitters"""

    text_splitter = RecursiveCharacterTextSplitter( chunk_size = chunk_size, chunk_overlap = chunk_overlap, add_start_index = True)
    texts = text_splitter.split_documents(document)

    return texts





def Tokenizer_Text_Splitter(document, chunk_size, chunk_overlap):
    """Split text using Tokens via HuggingFace's Transformer tokenizer"""

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    text_splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer, chunk_size= chunk_size, chunk_overlap= chunk_overlap
)
    text = text_splitter.split_documents(document)
    return text




def load_vectorstore(save_path, embedding_model):
    """Load existing vectorstores"""

    vectorstore = FAISS.load_local(
        save_path,
        embedding_model,
        allow_dangerous_deserialization = True
    )
    return vectorstore







def main():
    load_dotenv()
    embedding_model_name = os.getenv("EMBEDDING_MODEL")
    llm = os.getenv("OLLAMA_LLM")


    pdf_folder_path = "./data"
    save_path = "FAISS"
    docs = load_docs(pdf_folder_path)



    text = Recursive_Text_Splitter(docs, 512, 30)



    embedding_model = OllamaEmbeddings(model = embedding_model_name)




    if os.path.exists(save_path):
        vectorstore = load_vectorstore(save_path, embedding_model)
    else:
        vectorstore = FAISS.from_documents(text, embedding_model)
        vectorstore.save_local(save_path)



    template = """
    Answer the question based on the following context:
    {context}

    Question : {question}

    INSTRUCTIONS:
    Answer the users QUESTION using the DOCUMENT text above.
    Keep your answer ground in the facts of the DOCUMENT.
    If the DOCUMENT doesn’t contain the facts to answer the QUESTION return NONE
"""



    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=llm)



    while True:
        print("Enter your question: \n(NOTE: ENTER 'q' TO QUIT)")
        ques = input()
        if ques == "q":
            break
        chain = prompt | model
        results = vectorstore.similarity_search(query = ques, k = 5)
        res = chain.invoke({"context": results, "question" : ques})
        print(res)
        
     
    




    


if __name__ == '__main__':
    main()

#TODO: Bug fixes related to metadata in pdf i.e: unable to answer the author of papers