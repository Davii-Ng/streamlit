import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader




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




def text_embedding():
    """Generate embeddings from """
    return


def main():

    pdf_folder_path = "./data"
    docs = load_docs(pdf_folder_path)

    


if __name__ == '__main__':
    main()