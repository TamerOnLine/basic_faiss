import os
import pandas as pd
import logging
from langchain.agents import AgentType, initialize_agent
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the LLM model
llm = OllamaLLM(model="llama3.2")
embeddings = OllamaEmbeddings(model="locusai/multi-qa-minilm-l6-cos-v1")

# Path to the FAISS database
FAISS_PATH = "faiss_chat_index"

# Load or create FAISS database
def initialize_faiss():
    """Initialize or load the FAISS database."""
    if os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):
        logging.info("Loading FAISS database...")
        return FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    logging.info("FAISS database not found. Creating a new one...")
    faiss_index = FAISS.from_texts(["Welcome to the chatbot!"], embeddings)
    faiss_index.save_local(FAISS_PATH)
    logging.info("New FAISS database created and saved.")
    return faiss_index

faiss_index = initialize_faiss()


def search_chat(query: str, k: int = 3, threshold: float = 0.7):
    """Search the FAISS database and return the best-matching response."""
    results = faiss_index.similarity_search_with_score(query, k)
    
    for doc, score in results:
        if "→ Bot: " in doc.page_content and score >= threshold:
            stored_question, stored_answer = doc.page_content.split("→ Bot: ")
            if query.lower().strip() in stored_question.lower().strip():
                return stored_answer
    return None


def store_chat(query: str, response: str):
    """Store a question-response pair in the FAISS database."""
    global faiss_index
    
    if search_chat(query, k=1, threshold=0.9):
        logging.warning("This question already exists in FAISS.")
        return
    
    formatted_text = f"User: {query} → Bot: {response}"
    faiss_index.add_texts([formatted_text])
    faiss_index.save_local(FAISS_PATH)


def load_data_from_file(file_path: str):
    """Load data from a CSV or TXT file into the FAISS database."""
    global faiss_index
    
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        texts = df.iloc[:, 0].tolist()
    elif file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as file:
            texts = file.readlines()
    else:
        logging.error("Unsupported file format! Use CSV or TXT.")
        return
    
    logging.info(f"Loading {len(texts)} entries into FAISS...")
    faiss_index.add_texts(texts)
    faiss_index.save_local(FAISS_PATH)
    logging.info("Data successfully added to FAISS.")


def split_text_into_chunks(text, chunk_size=200, overlap=50):
    """Split large texts into smaller chunks with overlap."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]


def store_large_text(query: str, response: str, chunk_size=200, overlap=50):
    """Store long responses in FAISS using sliding window chunks."""
    global faiss_index  
    
    chunks = split_text_into_chunks(response, chunk_size, overlap)
    chunked_texts = [f"User: {query} → Bot: {chunk}" for chunk in chunks]
    
    faiss_index.add_texts(chunked_texts)
    faiss_index.save_local(FAISS_PATH)
    logging.info(f"Stored {len(chunked_texts)} chunks in FAISS.")


def search_with_sliding_window(query: str, k=5, threshold=0.65):
    """Search FAISS with sliding window and return relevant results."""
    global faiss_index
    results = faiss_index.similarity_search_with_score(query, k)
    
    filtered_responses = [doc.page_content.split("→ Bot: ")[1] for doc, score in results if "→ Bot: " in doc.page_content and score >= threshold]
    
    return "\n---\n".join(filtered_responses[:3]) if filtered_responses else "No accurate results found."


# Define the FAISS search tool
search_tool = Tool(
    name="FAISS Search",
    func=search_chat,
    description="Searches stored chat history in FAISS to find relevant responses."
)

llm_tool = Tool(
    name="Ollama LLM",
    func=lambda query: llm.invoke(query),
    description="Uses Ollama LLM to answer queries when no response is found in FAISS."
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=[search_tool, llm_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)


def chat_with_faiss():
    """Run an interactive chat session using FAISS."""
    logging.info("Starting FAISS-powered chatbot...")
    print("Type 'exit' to leave the chat.")
    
    while True:
        query = input("You: ").strip()
        if query.lower() == "exit":
            break
        
        response = search_with_sliding_window(query)
        if response == "No accurate results found.":
            logging.info("Using Ollama LLM...")
            response = llm.invoke(query)
            store_large_text(query, response)
        
        print(f"Bot: {response}")


if __name__ == "__main__":
    chat_with_faiss()
