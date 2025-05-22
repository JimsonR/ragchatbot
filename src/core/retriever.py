from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


def create_retriever(vector_store, top_k=4):
    """Create a retriever with Gemini 2.0 Flash compression."""
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    
    try:
        # Initialize Gemini 2.0 Flash
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",  # or "models/gemini-1.5-flash-latest"
            temperature=0,
            convert_system_message_to_human=True
        )
        
        compressor = LLMChainExtractor.from_llm(llm)
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
    except Exception as e:
        print(f"⚠️ Compression failed (falling back to standard retriever): {e}")
        return base_retriever
