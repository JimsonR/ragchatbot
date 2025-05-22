import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from src.core.document_loader import load_documents
from src.core.text_processing import split_documents, create_embeddings
from src.core.vector_store import create_vector_store, load_vector_store
from src.core.retriever import create_retriever
from src.core.generator import create_llm, create_prompt_template
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Gemini RAG Chatbot", layout="wide")
st.title("🧠 Gemini-powered Document Chatbot")

st.subheader("🧾 Chat History")



def format_chat_history(history):
    formatted = ""
    for user_msg, bot_msg in history:
        formatted += f"User: {user_msg}\nAI: {bot_msg}\n"
    return formatted
# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "llm" not in st.session_state:
    st.session_state.llm = create_llm()
if "prompt_template" not in st.session_state:
    st.session_state.prompt_template = create_prompt_template()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(message["user"])
    with st.chat_message("assistant"):
        st.markdown(message["bot"])
# Load embeddings
embeddings = create_embeddings()

persist_dir = "./faiss_index"  # Or whatever your vector store persist directory is

def refresh_vector_store():
    """Delete existing vector store directory if exists, then rebuild vector store."""
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)  # delete existing persisted vectors
    documents = load_documents()
    if not documents:
        st.error("⚠️ No valid documents found in the `data/` folder.")
        return None
    chunks = split_documents(documents)
    if not chunks:
        st.error("⚠️ Document splitting returned no chunks. Please check content.")
        return None
    vector_store = create_vector_store(chunks, embeddings)
    st.success("✅ Created and saved new vector store.")
    return vector_store

# Load or create vector store on app start
if st.session_state.vector_store is None:
    with st.spinner("🔍 Initializing vector store..."):
        vector_store = load_vector_store(embeddings)
        if vector_store:
            st.success("✅ Loaded existing vector store.")
        else:
            vector_store = refresh_vector_store()
        st.session_state.vector_store = vector_store
        if vector_store:
            st.session_state.retriever = create_retriever(vector_store)

# Add a button to refresh vector store manually (optional)
if st.button("🔄 Refresh Vector Store (Reload documents)"):
    with st.spinner("Refreshing vector store..."):
        vector_store = refresh_vector_store()
        if vector_store:
            st.session_state.vector_store = vector_store
            st.session_state.retriever = create_retriever(vector_store)

# Input for user query
if st.session_state.retriever:
    st.subheader("Ask a question about your documents:")
    query = st.text_input("Type your question:")

if query:
    with st.spinner("💬 Generating answer..."):
        qa_chain = RetrievalQA.from_chain_type(
    llm=st.session_state.llm,
    chain_type="stuff",
    retriever=st.session_state.retriever,
    chain_type_kwargs={"prompt": st.session_state.prompt_template},
    return_source_documents=True
)
        result = qa_chain({"query": query})
        answer = result["result"]

        # Append to chat history
        st.session_state.chat_history.append({"user": query, "bot": answer})

        # Show latest response
        st.subheader("💡 AI Response")
        st.write(answer)

        # Show source documents
        if result.get("source_documents"):
            st.subheader("📄 Source Documents")
            for i, doc in enumerate(result["source_documents"], 1):
                with st.expander(f"Source {i}: {doc.metadata.get('source', 'Unknown')}"):
                    st.write(doc.page_content)


# Sidebar info
with st.sidebar:
    st.markdown("### ℹ️ System Pipeline")
    st.code("""
Documents → Text Splitter → Embeddings → Vector Store (FAISS)
                          ↓
                   Retriever (Gemini)
                          ↓
               Prompt → Gemini LLM → Answer
    """)
    if st.session_state.vector_store:
        st.success("Vector store is ready!")



# from src.core.retriever import create_retriever
# from src.core.generator import create_llm, create_prompt_template
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from dotenv import load_dotenv
# import streamlit as st
# import os

# # Initialize Streamlit
# st.set_page_config(page_title="RAG with ChromaDB", layout="wide")
# st.title("🔍 Document Q&A with ChromaDB")

# # Load environment variables
# load_dotenv()

# # Initialize session state
# if 'vector_db' not in st.session_state:
#     st.session_state.vector_db = None
# if 'retriever' not in st.session_state:
#     st.session_state.retriever = None
# if 'llm' not in st.session_state:
#     st.session_state.llm = create_llm()
# if 'prompt_template' not in st.session_state:
#     st.session_state.prompt_template = create_prompt_template()

# def process_files(uploaded_files):
#     """Convert files to LangChain Documents"""
#     documents = []
#     for file in uploaded_files:
#         if file.name.endswith('.txt'):
#             try:
#                 text = file.read().decode('utf-8')
#                 documents.append(Document(
#                     page_content=text,
#                     metadata={"source": file.name}
#                 ))
#             except Exception as e:
#                 st.warning(f"Skipped {file.name}: {str(e)}")
#     return documents

# def initialize_chroma(documents):
#     """Initialize ChromaDB vector store"""
#     # Split documents
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     splits = text_splitter.split_documents(documents)
    
#     # Create embeddings
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
#     # Initialize ChromaDB (persists to disk by default)
#     vector_db = Chroma.from_documents(
#         documents=splits,
#         embedding=embeddings,
#         persist_directory="./chroma_db"  # Stores embeddings locally
#     )
    
#     # Create retriever using your custom function
#     retriever = create_retriever(vector_db)
    
#     return vector_db, retriever

# # File uploader
# uploaded_files = st.file_uploader(
#     "Upload text files", 
#     type=['txt'],
#     accept_multiple_files=True
# )

# if uploaded_files and not st.session_state.vector_db:
#     with st.spinner("Indexing documents..."):
#         documents = process_files(uploaded_files)
#         if documents:
#             st.session_state.vector_db, st.session_state.retriever = initialize_chroma(documents)
#             st.success(f"Loaded {len(documents)} documents into ChromaDB!")

# # Query interface
# if st.session_state.retriever:
#     query = st.text_input("Ask a question:")
    
#     if query:
#         with st.spinner("Searching ChromaDB..."):
#             try:
#                 # 1. Retrieve from ChromaDB
#                 docs = st.session_state.retriever.get_relevant_documents(query)
                
#                 # 2. Display context
#                 st.subheader("📄 Retrieved Context")
#                 for doc in docs:
#                     with st.expander(f"From {doc.metadata['source']}"):
#                         st.write(doc.page_content)
                
#                 # 3. Generate answer
#                 context = "\n\n".join(d.page_content for d in docs)
#                 prompt = st.session_state.prompt_template.format_prompt(
#                     context=context,
#                     question=query
#                 )
#                 response = st.session_state.llm(prompt.to_messages())
                
#                 st.subheader("💡 AI Response")
#                 st.write(response.content)
                
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

# # Sidebar info
# with st.sidebar:
#     st.markdown("### System Architecture")
#     st.code("""
#     Documents → Text Splitting → Embeddings → ChromaDB
#                       ↓
#                Your retriever.py → Contextual Compression
#                       ↓
#                Your generator.py → LLM Response
#     """)
#     if st.session_state.vector_db:
#         st.success(f"ChromaDB contains {st.session_state.vector_db._collection.count()} vectors")


# from src.core.retriever import create_retriever
# from src.core.generator import create_llm, create_prompt_template
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from dotenv import load_dotenv
# import streamlit as st
# import os

# # Initialize Streamlit
# st.set_page_config(page_title="RAG with Vector DB", layout="wide")
# st.title("🔍 Document Q&A with VectorDB Retrieval")

# # Load environment variables
# load_dotenv()

# # Initialize session state
# if 'vector_db' not in st.session_state:
#     st.session_state.vector_db = None
# if 'retriever' not in st.session_state:
#     st.session_state.retriever = None
# if 'llm' not in st.session_state:
#     st.session_state.llm = create_llm()  # From generator.py
# if 'prompt_template' not in st.session_state:
#     st.session_state.prompt_template = create_prompt_template()  # From generator.py

# # Document processing
# def process_files(uploaded_files):
#     """Convert uploaded files to LangChain Documents"""
#     documents = []
#     for file in uploaded_files:
#         if file.name.endswith('.txt'):
#             try:
#                 text = file.read().decode('utf-8')
#                 documents.append(Document(
#                     page_content=text,
#                     metadata={"source": file.name}
#                 ))
#             except Exception as e:
#                 st.warning(f"Skipped {file.name}: {str(e)}")
#     return documents

# # Initialize VectorDB and Retriever
# def initialize_vectordb(documents):
#     """Create FAISS vector store and retriever"""
#     # Split documents
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     splits = text_splitter.split_documents(documents)
    
#     # Create embeddings and vector store
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vector_db = FAISS.from_documents(splits, embeddings)
    
#     # Initialize retriever (using your retriever.py)
#     retriever = create_retriever(vector_db) 
    
#     return vector_db, retriever

# # File uploader
# uploaded_files = st.file_uploader(
#     "Upload text files", 
#     type=['txt'],
#     accept_multiple_files=True
# )

# if uploaded_files and not st.session_state.vector_db:
#     with st.spinner("Indexing documents..."):
#         documents = process_files(uploaded_files)
#         if documents:
#             st.session_state.vector_db, st.session_state.retriever = initialize_vectordb(documents)
#             st.success(f"Loaded {len(documents)} documents into VectorDB!")

# # Query interface
# if st.session_state.retriever:
#     query = st.text_input("Ask a question:")
    
#     if query:
#         with st.spinner("Searching VectorDB..."):
#             try:
#                 # 1. Retrieve from VectorDB
#                 docs = st.session_state.retriever.get_relevant_documents(query)
                
#                 # 2. Display context
#                 st.subheader("📄 Retrieved Context")
#                 for doc in docs:
#                     with st.expander(f"From {doc.metadata['source']}"):
#                         st.write(doc.page_content)
                
#                 # 3. Generate answer (using generator.py)
#                 context = "\n\n".join(d.page_content for d in docs)
#                 prompt = st.session_state.prompt_template.format_prompt(
#                     context=context,
#                     question=query
#                 )
#                 response = st.session_state.llm(prompt.to_messages())
                
#                 st.subheader("💡 AI Response")
#                 st.write(response.content)
                
#             except Exception as e:
#                 st.error(f"Error: {str(e)}")

# # Sidebar info
# with st.sidebar:
#     st.markdown("### System Architecture")
#     st.code("""
#     Documents → Text Splitting → Embeddings → FAISS VectorDB
#                       ↓
#                Your retriever.py → Contextual Compression
#                       ↓
#                Your generator.py → LLM Response
#     """)

# from langchain.retrievers import ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import LLMChainExtractor
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.documents import Document
# from langchain_core.runnables import RunnableLambda
# from dotenv import load_dotenv
# import streamlit as st
# import os

# # Initialize Streamlit app
# st.set_page_config(page_title="Document RAG Chatbot", layout="wide")
# st.title("📄 Document Analysis Chatbot")
# st.write("Upload and query your text documents")

# # Load environment variables
# load_dotenv()

# # Initialize session state
# if 'vector_store' not in st.session_state:
#     st.session_state.vector_store = None
# if 'retriever' not in st.session_state:
#     st.session_state.retriever = None

# # Document processing function
# def process_uploaded_files(uploaded_files):
#     """Process uploaded text files into documents"""
#     documents = []
#     for file in uploaded_files:
#         if file.name.endswith('.txt'):
#             try:
#                 text = file.read().decode('utf-8')
#                 documents.append(Document(
#                     page_content=text,
#                     metadata={"source": file.name}
#                 ))
#             except Exception as e:
#                 st.warning(f"Couldn't process {file.name}: {str(e)}")
#     return documents

# # Initialize retriever
# def create_retriever(documents):
#     """Create vector store and retriever from documents"""
#     # Split documents
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     splits = text_splitter.split_documents(documents)
    
#     # Create embeddings and vector store
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#     vector_store = FAISS.from_documents(splits, embeddings)
    
#     # Create retriever function
#     def retrieve_docs(query: str):
#         return vector_store.similarity_search(query, k=4)
    
#     # Convert to Runnable
#     base_retriever = RunnableLambda(retrieve_docs)
    
#     try:
#         llm = ChatGoogleGenerativeAI(
#             model="gemini-1.5-flash",
#             temperature=0.3
#         )
#         compressor = LLMChainExtractor.from_llm(llm)
#         return ContextualCompressionRetriever(
#             base_compressor=compressor,
#             base_retriever=base_retriever
#         ), vector_store
#     except Exception as e:
#         st.warning(f"⚠️ Compression disabled: {str(e)}")
#         return base_retriever, vector_store

# # File upload section
# uploaded_files = st.file_uploader(
#     "Upload text files",
#     type=['txt'],
#     accept_multiple_files=True,
#     help="Upload .txt files to build your knowledge base"
# )

# if uploaded_files and st.session_state.vector_store is None:
#     with st.spinner("Processing documents (this only happens once)..."):
#         documents = process_uploaded_files(uploaded_files)
#         if documents:
#             st.session_state.retriever, st.session_state.vector_store = create_retriever(documents)
#             st.success(f"Processed {len(documents)} documents! You can now ask questions.")
#         else:
#             st.error("No valid documents were processed")

# if st.session_state.retriever:
#     # Chat interface
#     query = st.text_input("Ask about your documents:", placeholder="Type your question here...")
    
#     if query:
#         with st.spinner("Searching documents..."):
#             try:
#                 docs = st.session_state.retriever.invoke(query)
                
#                 st.subheader("🔍 Relevant Document Sections")
#                 for i, doc in enumerate(docs, 1):
#                     with st.expander(f"From {doc.metadata['source']}"):
#                         st.write(doc.page_content)
                
#                 # Generate comprehensive answer
#                 llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
#                 context = "\n\n---\n\n".join(
#                     f"SOURCE: {doc.metadata['source']}\nCONTENT:\n{doc.page_content}" 
#                     for doc in docs
#                 )
                
#                 response = llm.invoke(f"""You're a professional document analyst. 
#                 Based on these document sections:
#                 {context}
                
#                 Question: {query}
                
#                 Provide a detailed answer with:
#                 1. Key points from the documents
#                 2. Any relevant details
#                 3. Source references when applicable
#                 Answer:""")
                
#                 st.subheader("📝 Analysis Summary")
#                 st.write(response.content)
                
#             except Exception as e:
#                 st.error(f"Analysis failed: {str(e)}")
# elif not uploaded_files:
#     st.info("Please upload text files to build your knowledge base")

# # Sidebar with info
# with st.sidebar:
#     st.markdown("### System Information")
#     st.markdown("""
#     - **Embedding Model**: all-MiniLM-L6-v2
#     - **LLM**: Gemini 1.5 Flash
#     - **Chunk Size**: 1000 characters
#     - **Overlap**: 200 characters
#     """)
#     if st.session_state.vector_store:
#         st.success("✅ Documents are loaded and ready")
#     st.markdown("💡 Tip: For best results, upload multiple related documents")
