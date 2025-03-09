import os
import logging
import requests
from bs4 import BeautifulSoup
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Ollama LLM
llm = OllamaLLM(model="llama3.2")

# Initialize text-to-vector model
embeddings = OllamaEmbeddings(model="locusai/multi-qa-minilm-l6-cos-v1")

# FAISS database path
FAISS_PATH = "faiss_chat_index"

def initialize_faiss():
    """
    Load or create a FAISS database.

    Returns:
        FAISS: The FAISS index object.
    """
    faiss_file = os.path.join(FAISS_PATH, "index.faiss")
    if os.path.exists(faiss_file):
        logging.info("Loading FAISS database...")
        return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    logging.info("FAISS database not found, creating a new one...")
    sample_texts = [
        "Hello, how can I help you?", 
        "Welcome to FAISS!", 
        "This is a test entry."
    ]

    faiss_index = FAISS.from_texts(sample_texts, embeddings)
    faiss_index.save_local(FAISS_PATH)
    logging.info("FAISS database created successfully.")
    return faiss_index

faiss_index = initialize_faiss()

def classify_question(query: str) -> str:
    """
    Classify the question using Ollama before directing it to the appropriate tool.
    
    Args:
        query (str): The user query.
    
    Returns:
        str: Either 'faiss_search' for database searches or 'ollama_ai' for AI-generated answers.
    """
    classification_prompt = (
        "Classify the question into one of these categories:\n"
        "1. Search FAISS database\n"
        "2. Generate a new response with Ollama\n"
        f"\nQuestion: {query}"
    )
    classification = llm.invoke(classification_prompt).strip().lower()
    
    return "faiss_search" if "faiss" in classification else "ollama_ai"

def scrape_webpage(url: str) -> str:
    """
    Extract text content from a webpage.
    
    Args:
        url (str): The URL of the webpage to scrape.
    
    Returns:
        str: Extracted text content, truncated to 5000 characters if necessary.
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    paragraphs = soup.find_all("p")
    text_content = "\n".join([p.get_text() for p in paragraphs if p.get_text()])

    return text_content[:5000] + "..." if len(text_content) > 5000 else text_content

# Define web scraper tool
web_scraper_tool = Tool(
    name="WebScraper",
    func=scrape_webpage,
    description="Scrapes webpage content and extracts text data.",
    return_direct=True,
)

def search_faiss(query: str, k: int = 5, threshold: float = 0.6) -> str:
    """
    Perform a similarity search in FAISS and return relevant results.

    Args:
        query (str): The search query.
        k (int, optional): Number of results to retrieve. Defaults to 5.
        threshold (float, optional): Minimum similarity score. Defaults to 0.6.

    Returns:
        str: A string of search results or a message if none are found.
    """
    results = faiss_index.similarity_search_with_score(query, k=k)
    filtered_responses = [
        doc.page_content.split("\u2192 Bot: ")[1]
        for doc, score in results if "\u2192 Bot: " in doc.page_content and score >= threshold
    ]
    return "\n---\n".join(filtered_responses[:3]) if filtered_responses else "No relevant results found."

def ollama_fallback(query: str) -> str:
    """
    Use Ollama to generate an answer if FAISS has no relevant responses.
    
    Args:
        query (str): The user query.
    
    Returns:
        str: The generated response or an error message.
    """
    try:
        return llm.invoke(query) or "No answer found."
    except Exception as e:
        logging.error(f"Error querying Ollama: {e}")
        return "An error occurred while retrieving an answer."

def execute_tool(tool_name: str, query: str) -> str:
    """
    Execute the appropriate tool based on classification.

    Args:
        tool_name (str): The tool to use ('faiss_search' or 'ollama_ai').
        query (str): The user query.

    Returns:
        str: The tool's response.
    """
    if tool_name == "faiss_search":
        result = search_faiss(query)
        return result if result != "No relevant results found." else ollama_fallback(query)
    
    elif tool_name == "ollama_ai":
        return ollama_fallback(query)

    return "Unknown tool, please check your input."

# Define search tools
search_tool = Tool(name="faiss_search", func=search_faiss, description="Search FAISS database for relevant answers.")
ollama_tool = Tool(name="ollama_ai", func=ollama_fallback, description="Uses Ollama LLM for question answering.")

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent
agent = initialize_agent(
    tools=[search_tool, ollama_tool, web_scraper_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    max_iterations=10,
    handle_parsing_errors=True
)

def process(query: str) -> str:
    """
    Run the agent to process queries.
    
    Args:
        query (str): The user query.
    
    Returns:
        str: The response generated by the agent.
    """
    logging.info(f"Processing query: {query}")
    result = agent.invoke(query)
    return result.get("output", "No response")

def chat():
    """
    Start an interactive chat session using FAISS and Ollama.
    """
    logging.info("Starting chat session...")
    print("Type 'exit' to end the conversation.")
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() == "exit":
                break
            category = classify_question(query)
            response = execute_tool(category, query)
            print(f"Bot: {response}")
        except Exception as e:
            logging.error(f"Chat error: {e}")
            print("An error occurred, please try again.")

if __name__ == "__main__":
    chat()
