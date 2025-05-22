# src/core/vector_store.py
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
import os

CHROMA_DIR = "chroma_db"

def create_vector_store(text_chunks: list[Document], embeddings: Embeddings):
    vector_store = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vector_store.persist()
    return vector_store

def load_vector_store(embeddings: Embeddings):
    if not os.path.exists(CHROMA_DIR):
        return None
    try:
        return Chroma(
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR
        )
    except Exception:
        return None
