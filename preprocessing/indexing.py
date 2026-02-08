import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import (
    HuggingFaceEmbeddings,
)  # Free alternative for now! ( works like a charm )

import os
from dotenv import load_dotenv

load_dotenv()


def index_clinical_trials(
    json_data_path: str,
    output_path: str,
    embedding_model: str = "models/gemini-embedding-001",
) -> None:
    """
    Indexes clinical trial data into a Chroma vector store.

    Loads JSON data containing text_data and meta_data for each patient,
    converts them into LangChain Documents (page_content=text_data, metadata=meta_data),
    and stores them in a Chroma vector database for later retrieval.

    Args:
        json_data_path: Path to the JSON file containing preprocessed clinical trial data
        output_path: Directory path where the Chroma vector store will be persisted
        embedding_model: Gemini embedding model to use (default: models/text-embedding-004)

    Returns:
        None. Creates and persists a Chroma vector store at output_path.

    Example:
        >>> index_clinical_trials(
        ...     json_data_path='processed_data.json',
        ...     output_path='./chroma_db'
        ... )
    """
    # Load the JSON data
    with open(json_data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records from {json_data_path}")

    # Initialize Gemini embeddings
    # text-embedding-004 is optimized for retrieval tasks and works best with Gemini models
    # embeddings = GoogleGenerativeAIEmbeddings(model=embedding_model, api_key=os.getenv("GOOGLE_API_KEY"))

    # using this because it is free
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Convert each record into a LangChain Document
    documents = []
    for idx, record in enumerate(data):
        text_data = record.get("text_data", {})
        meta_data = record.get("meta_data", {})

        # Combine all text_data values into page_content
        # This creates a rich text representation for embedding
        text_parts = []
        for key, value in text_data.items():
            if value:  # Only include non-empty values
                text_parts.append(f"{key}: {value}")

        page_content = "\n".join(text_parts)

        # Create Document with text as page_content and meta_data as metadata
        # LangChain will embed page_content and keep metadata for filtering/querying
        doc = Document(page_content=page_content, metadata=meta_data)
        documents.append(doc)

    print(f"Created {len(documents)} documents for indexing")

    # Create and persist the Chroma vector store
    vector_store = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=output_path
    )

    print(f"Successfully indexed {len(documents)} documents to {output_path}")
    print(f"Vector store is ready for retrieval!")


if __name__ == "__main__":
    # Example usage
    index_clinical_trials(
        json_data_path="/Users/mac/Desktop/Projects/clinical_trials/data/clean_data.json",
        output_path="/Users/mac/Desktop/Projects/clinical_trials/data/chroma_clinical_trials",
    )
