from dotenv import load_dotenv
from langchain_google_community import GoogleSearchAPIWrapper
from typing import List, Tuple

# Load environment variables from a .env file
load_dotenv()
def google_web_search(question: str, k: int) -> Tuple[List[str], List[str]]:
    """
    Perform a web search using Google and retrieve relevant content and links.

    Args:
        question (str): The input question.
        k (int): The number of results to return.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing lists of links and contents.
    """
    # Initialize Google Search API Wrapper
    search = GoogleSearchAPIWrapper()

    try:
        # Perform search and get the results
        results = search.results(query=question, num_results=k)

        # Extract URLs and snippets
        google_urls = [result['link'] for result in results]
        google_contents = [result['snippet'] for result in results if 'snippet' in result]
        
        print("\nGoogle urls:", google_urls, "\nGoogle contents:", google_contents)
        return google_urls, google_contents

    except Exception as e:
        print(f"Error fetching Google results: {e}")
        return [], []  # Return empty lists in case of an error
