from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

def create_llm(model_name="gemini-2.0-flash", temperature=0.7):
    # Configure Gemini
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Create LangChain wrapper
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        convert_system_message_to_human=True  # Gemini-specific setting
    )

def create_prompt_template():
    template = """
    You are a helpful AI assistant. Use the following context to answer the question.
    If you don't know, say you don't know. Never make up answers.
    
    Context: {context}
    
    Question: {question}
    
    Answer: """
    
    return ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a helpful AI assistant."),
        HumanMessagePromptTemplate.from_template(template)
    ])
