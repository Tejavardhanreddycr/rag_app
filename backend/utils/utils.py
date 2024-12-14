import os
from typing import List, Any, Dict
from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import cohere
import time
from dotenv import load_dotenv
from utils.llms import get_cohere_llm, get_openai_llm

# Load environment variables from a .env file
load_dotenv()

# Retrieve API keys from environment variables
cohere_api_key = os.getenv('COHERE_API_KEY')
pinecone_api_key = os.getenv('PINECONE_API_KEY')
index_name = 'rag-data'

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "amd-convogene"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "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"


def create_chains():
    gpt_model = get_openai_llm()

    prompt = """You are to answer the question based on the model. 
        You need to answer the question only based on the given context 
        provided below and not make up the answer on your own.
        Don't include previous history \n
        Context: {context} \n
        Question: {question}
        """
    prompt = PromptTemplate.from_template(prompt)

    gpt_chain = (
            prompt
            | gpt_model
            | StrOutputParser()
    )
    
    
    return gpt_chain


def get_sources(docs):
    sources = []
    for doc in docs:
        sources.append(doc.metadata["source"] + "\n")
    # print("SOURCE")
    # print(sources)
    return sources[:3]

def grading_1(data, query):
    try:
        print("Grading in progress")
        approved_docs = []
        cohere_model = get_cohere_llm()
        for doc in data:
            grade_prompt = f"""Instruction: Determine if the following text contains the answer to the given question based on context.

                            Question: {query}

                            Context: {str(doc.page_content)}

                            Answer: Does the text contain the answer to the question? Please respond with "Yes" or "No".
                            
                            FOLLOW THE ABOVE COMMANDS STRICTLY
                            """
            response = cohere_model.invoke(grade_prompt)
        
            if response.content == "Yes":
                print("APPROVED")
                approved_docs.append(doc)
            else:
                print("REJECTED")
        return approved_docs
    except Exception as e:
        print("error in grading:",e)


def answer(relevant_docs: List[str], question: str) -> str:
    """
    Generate an answer based on relevant documents and the question.

    Args:
        relevent_docs (List[str]): A list of relevant documents.
        question (str): The question to be answered.

    Returns:
        str: The generated response.
    """
    try:
        gpt_model = get_openai_llm()
        # relevant_docs = format_docs(relevant_docs)
        prompt = f"""You are to answer the question based on the model. 
        You need to answer the question only based on the given context 
        provided below and not make up the answer on your own. \n
        Context: {relevant_docs} \n
        Question: {question}
        """
        print("prompt")
        gpt_response = gpt_model.invoke(prompt)
        print(gpt_response.content)
        return gpt_response
    
    except Exception as e:
        print(f"Error in generating answer: {e}")
        return "An error occurred while generating the answer."


# Cohere Reranking Function (from Code 1)
def rerank_docs(query: str, docs, top_n: int = 5):
    try:
        cohere_client = cohere.Client(api_key=os.getenv('COHERE_API_KEY'))
        texts = [doc.page_content for doc in docs]
        reranked = cohere_client.rerank(
            query=query,
            documents=texts,
            top_n=top_n,
            model="rerank-english-v3.0"
        )
        # Access the reranked indices using 'results' attribute
        return [docs[result.index] for result in reranked.results]
    except Exception as e:
        print("error in reranking:",e)

def get_vectorstore() -> PineconeVectorStore:
    """
    Initialize and return the Pinecone vector store with Cohere embeddings.

    Returns:
        PineconeVectorStore: The initialized Pinecone vector store.
    """
    try:
        if not cohere_api_key:
            raise ValueError("Cohere API key not found.")
        if not pinecone_api_key:
            raise ValueError("Pinecone API key not found.")
        
        # Initialize Cohere embeddings
        embed = CohereEmbeddings(cohere_api_key=cohere_api_key, model="embed-english-v2.0")
        
        # Initialize Pinecone vector store with embeddings
        vector_store = PineconeVectorStore(
            pinecone_api_key=pinecone_api_key,
            index_name=index_name,
            embedding=embed,
        )
        return vector_store
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        raise

import concurrent.futures

def grading(data, query):
    try:
        print("Grading in progress")
        approved_docs = []
        cohere_model = get_cohere_llm()

        def grade_single_doc(doc):
            grade_prompt = f"""Instruction: Determine if the following text contains the exact answer to the given question based on context

                            Question: {query}

                            Context: {str(doc.page_content)}

                            Answer: Does the text contain the answer to the question? Please respond with "Yes" or "No".
                            
                            FOLLOW THE ABOVE COMMANDS STRICTLY
                            """
            response = cohere_model.invoke(grade_prompt)
            return (doc, response.content == "Yes")  # Return the doc and whether it's approved or not

        # Use ThreadPoolExecutor to grade documents concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_doc = {executor.submit(grade_single_doc, doc): doc for doc in data}

            for future in concurrent.futures.as_completed(future_to_doc):
                doc, approved = future.result()
                if approved:
                    print("APPROVED")
                    approved_docs.append(doc)
                else:
                    print("REJECTED")

        return approved_docs

    except Exception as e:
        print("error in grading:", e)

# def format_docs(docs: List[Any]) -> str:
#     """
#     Format a list of documents into a single string with each document's content separated by two newlines.

#     Args:
#         docs (List[Any]): The list of documents.

#     Returns:
#         str: The formatted string of documents.
#     """
#     try:
#         return '\n\n'.join(doc.page_content for doc in docs)
#     except Exception as e:
#         print(f"Error formatting documents: {e}")
#         return ""


def format_web_docs(docs: List[Dict[str, Any]]) -> str:
    """
    Format a list of document strings into a single string with each document's content separated by two newlines.

    Args:
        docs (List[str]): The list of document strings.

    Returns:
        str: The formatted string of documents.
    """
    try:
        return '\n\n'.join(docs)
    except Exception as e:
        print(f"Error formatting documents: {e}")
        return ""
        
def question_reframe(question: str) -> str:
    """
    Reframe the input question to make it more precise and actionable.

    Args:
        question (str): The input question.

    Returns:
        str: The reframed question.
    """
    try:
        model = get_cohere_llm()
        PROMPT = f"""
        Rephrase the question to make it more precise and actionable
        
        {question}
        
        Output only the reframed question:
        """
        reframed_prompt = model.invoke(PROMPT)
        reframed_question = reframed_prompt.content.strip()
        return reframed_question
    except Exception as e:
        print(f"Error reframing question: {e}")
        return question


def format_docs(docs: List[Any]) -> str:
    """
    Format a list of documents into a single string with each document's content and metadata separated by two newlines.

    Args:
        docs (List[Any]): The list of documents. Each document is expected to have 'page_content' and 'metadata' attributes.

    Returns:
        str: The formatted string of documents with metadata.
    """
    try:
        formatted_content = '\n\n'.join(doc.page_content for doc in docs)
        
        # Extract metadata for each document
        metadata = [doc.metadata['source'] for doc in docs]
        return formatted_content,metadata
        
    except Exception as e:
        print(f"Error formatting documents: {e}")
        return ""

def stream_response(links):
    # print(sorted(links))
    for link in sorted(links,reverse=True):
        # print("links:",link)
        for ch in link:
            yield ch
            time.sleep(0.001)
        yield "\n"
