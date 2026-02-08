from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict
from pydantic import BaseModel, Field
from langchain.tools import tool


class ClinicalTrialRetrievalInput(BaseModel):
    """Input schema for clinical trial retrieval tool."""

    query: str = Field(description="The search query to find relevant clinical trials")


@tool(args_schema=ClinicalTrialRetrievalInput)
def clinical_trial_retrieval(
    query: str,
    chroma_dir: str = "/Users/mac/Desktop/Projects/clinical_trials/data/chroma_clinical_trials",
    k: int = 10,
) -> List[Dict]:
    """
    Performs semantic retrieval from a vector store of clinical trial data.

    Searches through indexed clinical trials using semantic similarity to find
    the most relevant trials based on the query. Returns both the text content
    and associated metadata (NCT Number, age, sex, condition, etc.).

    Args:
        query: The search query describing the clinical trial criteria
        chroma_dir: Directory path to the persisted Chroma vector store
        k: Number of top results to retrieve (default: 10)

    Returns:
        List of dictionaries containing retrieved content and metadata for each trial
    """
    # Initialize the same embedding model used during indexing
    # Using HuggingFace embeddings (free alternative)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Load the persisted Chroma vector store
    vector_store = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

    # Create retriever with similarity search
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": k}
    )

    # Perform retrieval
    full_results = retriever.invoke(query)

    # Format results to include both content and metadata
    formatted_results = []
    for result in full_results:
        formatted_results.append(
            {"content": result.page_content, "metadata": result.metadata}
        )

    return formatted_results
