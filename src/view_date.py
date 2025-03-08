import os
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# Path to the FAISS database
FAISS_PATH = "faiss_chat_index"

# Load the embedding model used for indexing
embeddings = OllamaEmbeddings(model="locusai/multi-qa-minilm-l6-cos-v1")


def list_all_stored_data():
    """
    Displays all stored data inside the FAISS database.

    This function checks if the FAISS database exists and retrieves 
    all stored data entries. If no data is found, it prints an appropriate message.
    """
    if not os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):
        print("FAISS database not found. Ensure data has been saved previously.")
        return

    # Load FAISS database
    faiss_index = FAISS.load_local(
        FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )

    # Retrieve all stored entries
    stored_texts = faiss_index.similarity_search(" ", k=1000)  # Fetch all data

    if not stored_texts:
        print("The FAISS database is empty! No data has been stored yet.")
        return

    print("All stored data in FAISS:")
    for i, doc in enumerate(stored_texts, start=1):
        print(f"{i}. {doc.page_content}")


# Run the function to display data
if __name__ == "__main__":
    list_all_stored_data()
