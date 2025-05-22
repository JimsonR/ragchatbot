# app.py or a separate script (e.g., run_chatbot.py)

from dotenv import load_dotenv
from src.core.document_loader import load_documents
from src.core.text_processing import split_documents, create_embeddings
from src.core.vector_store import create_vector_store, load_vector_store
from src.core.retriever import create_retriever
from src.core.generator import create_llm, create_prompt_template
from langchain.chains import RetrievalQA

load_dotenv()

class GeminiRAGChatbot:
    def __init__(self, persist_dir="chroma_db"):
        self.embeddings = create_embeddings()
        self.llm = create_llm()
        self.prompt = create_prompt_template()
        
        # Load or create vector store (Chroma)
        self.vector_store = load_vector_store(self.embeddings, persist_dir=persist_dir)
        
        if not self.vector_store:
            documents = load_documents()
            if not documents:
                raise ValueError("❌ No documents found in the data/ folder.")
            text_chunks = split_documents(documents)
            self.vector_store = create_vector_store(text_chunks, self.embeddings, persist_dir=persist_dir)
        
        self.retriever = create_retriever(self.vector_store)
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )
    
    def query(self, question):
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }

if __name__ == "__main__":
    print("Initializing Gemini RAG Chatbot with ChromaDB...")
    chatbot = GeminiRAGChatbot()
    
    print("System ready. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            break
        
        response = chatbot.query(user_input)
        print(f"\nBot: {response['answer']}")
        
        if response["sources"]:
            print("\nSources:")
            for doc in response["sources"]:
                print(f"- {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'N/A')})")
        print("")
