import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool
import language_tool_python
from langchain.agents import AgentType, initialize_agent
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the language tool for grammar and spelling correction
language_tool = language_tool_python.LanguageTool('en-US')

# Function to correct user input
def correct_input(user_input: str) -> str:
    """Correct grammar and spelling in the user's input."""
    matches = language_tool.check(user_input)
    corrected_input = language_tool_python.utils.correct(user_input, matches)
    return corrected_input

# Function to scrape webpage content
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

# Initialize language model (Ollama)
llm = OllamaLLM(model="llama3.2")

# Initialize agent with web scraper tool
agent = initialize_agent(
    [web_scraper_tool],
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)

# Initialize the prompt template for the webpage scraping task
prompt_template = PromptTemplate(
    input_variables=["url"],
    template="Scrape the webpage content at {url}."
)

# Use the agent to invoke web scraping and correct input
def start_scraping_session():
    """Start a conversation with the user for scraping tasks."""
    print('Hello! I can help you scrape and extract content from webpages.')
    
    while True:
        user_input = input('You: ').strip()
        
        # Exit condition
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print('Goodbye! Have a great day!')
            break
        
        # Correct user's input
        corrected_input = correct_input(user_input)
        
        if corrected_input != user_input:
            print(f'Bot: Did you mean: "{corrected_input}"? (Corrected for clarity)')
        
        # Invoke the agent for scraping based on corrected input (assuming user provides a URL)
        response = agent.invoke(corrected_input)
        print(f'Bot: {response}')

if __name__ == '__main__':
    start_scraping_session()
