import os
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from a .env file
load_dotenv()

# Retrieve API keys from environment variables
cohere_api_key = os.getenv('COHERE_API_KEY')
open_ai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')

def get_cohere_llm() -> Optional[ChatCohere]:
    """
    Initialize and return the Cohere LLM model.

    Returns:
        Optional[ChatCohere]: The Cohere LLM model or None if an error occurs.
    """
    try:
        if not cohere_api_key:
            raise ValueError("Cohere API key not found.")
        model = ChatCohere(model="command-r-plus", cohere_api_key=cohere_api_key,streaming=True)
        return model
    except Exception as e:
        print(f"Error initializing Cohere model: {e}")
        return None

def get_openai_llm(model_name: str = 'gpt-4o') -> Optional[ChatOpenAI]:
    """
    Initialize and return the OpenAI LLM model.

    Args:
        model_name (str): The name of the OpenAI model to use. Default is 'gpt-3.5-turbo'.

    Returns:
        Optional[ChatOpenAI]: The OpenAI LLM model or None if an error occurs.
    """
    try:
        if not open_ai_api_key:
            raise ValueError("OpenAI API key not found.")
        model = ChatOpenAI(model=model_name, api_key=open_ai_api_key,streaming=True,temperature=0.4)
        return model
    except Exception as e:
        print(f"Error initializing OpenAI model: {e}")
        return None
