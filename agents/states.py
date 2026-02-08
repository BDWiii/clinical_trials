from typing import List, Dict, TypedDict
from pydantic import BaseModel, Field


# ======================= Pydantic Validation Schemas =======================


class RetrievalQuery(BaseModel):
    """Schema for controlling LLM output when using the retrieval tool."""

    query: str = Field(description="The search query to find relevant clinical trials")
    k: int = Field(default=5, description="Number of patients/trials to retrieve")


class RetrievedContent(BaseModel):
    """Schema for validated retrieved content from the vector store."""

    content: str = Field(description="The text content of the clinical trial")
    metadata: Dict = Field(
        description="Metadata including NCT Number, age, sex, condition, etc."
    )


class ValidationOutput(BaseModel):
    """Schema for validation agent output."""

    validated_content: List[Dict] = Field(
        description="List of validated clinical trial records"
    )


class ExplanationOutput(BaseModel):
    """Schema for explanation agent output."""

    explanations: List[str] = Field(
        description="List of explanations, one for each retrieved patient/trial"
    )


class RerankedOutput(BaseModel):
    """Schema for reranker agent output."""

    reranked_content: List[Dict] = Field(
        description="List of clinical trials reranked by severity/priority"
    )


class EmailOutput(BaseModel):
    """Schema for email agent output."""

    email_subject: str = Field(description="Subject line of the email")
    email_body: str = Field(description="Body content of the email")


# ======================= TypedDict States =======================


class RetrieverState(TypedDict):
    """State for the retriever agent."""

    task: str  # User input
    node_name: str
    next_node: str
    retrieved_content: List[Dict]


class ValidationState(TypedDict):
    """State for the validation agent."""

    task: str  # User input
    node_name: str
    next_node: str
    retrieved_content: List[Dict]  # Input from retriever
    validated_content: List[Dict]  # Output after validation


class ExplanationState(TypedDict):
    """State for the explanation agent."""

    task: str  # User input
    node_name: str
    next_node: str
    retrieved_content: List[Dict]  # Input from validation agent
    explanations: List[str]  # Output: one explanation per patient/trial


class RerankerState(TypedDict):
    """State for the reranker agent."""

    task: str  # User input
    node_name: str
    next_node: str
    retrieved_content: List[Dict]  # Input from validation
    reranked_content: List[Dict]  # Output: reranked by severity


class EmailState(TypedDict):
    """State for the email agent."""

    task: str  # User input
    node_name: str
    next_node: str
    patient_data: Dict  # First patient from retrieved_content
    email_subject: str  # Generated email subject
    email_body: str  # Generated email body


class MainState(TypedDict):
    """Main state for the compiled clinical trial agent."""

    task: str  # User input
    node_name: str
    next_node: str
    retrieved_content: List[Dict]
    validated_content: List[Dict]
    reranked_content: List[Dict]
    explanations: List[str]
    hitl: str  # Human in the loop response
    email_subject: str
    email_body: str
    history: List[Dict]
    # Sub-agent states
    retriever_state: RetrieverState
    validation_state: ValidationState
    reranker_state: RerankerState
    explanation_state: ExplanationState
    email_state: EmailState


# ==================== Initialization ====================


def _initialize_state(Input: str) -> MainState:
    """Initialize the main state for a new clinical trial query."""
    return {
        "task": Input,
        "node_name": "",
        "next_node": "",
        "retrieved_content": [],
        "validated_content": [],
        "reranked_content": [],
        "explanations": [],
        "hitl": "",
        "email_subject": "",
        "email_body": "",
        "history": [],
        "retriever_state": {
            "task": "",
            "node_name": "",
            "next_node": "",
            "retrieved_content": [],
        },
        "validation_state": {
            "task": "",
            "node_name": "",
            "next_node": "",
            "retrieved_content": [],
            "validated_content": [],
        },
        "reranker_state": {
            "task": "",
            "node_name": "",
            "next_node": "",
            "retrieved_content": [],
            "reranked_content": [],
        },
        "explanation_state": {
            "task": "",
            "node_name": "",
            "next_node": "",
            "retrieved_content": [],
            "explanations": [],
        },
        "email_state": {
            "task": "",
            "node_name": "",
            "next_node": "",
            "patient_data": {},
            "email_subject": "",
            "email_body": "",
        },
    }
