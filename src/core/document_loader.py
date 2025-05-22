import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredHTMLLoader

def load_documents(data_dir="data"):
    documents = []
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if filename.endswith(".txt"):
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            except UnicodeDecodeError:
                # Try alternative encoding fallback
                loader = TextLoader(file_path, encoding="ISO-8859-1")
                documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif filename.endswith(".html"):
            loader = UnstructuredHTMLLoader(file_path)
            documents.extend(loader.load())
    return documents
