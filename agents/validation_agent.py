import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import yaml
import aiofiles

from agents.states import ValidationState, ValidationOutput

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationAgent:
    """
    Validates retrieved clinical trial content and outputs clean, validated data.
    """

    def __init__(self, llm):
        """
        Initialize the validation agent.

        Args:
            llm: Language model instance (placeholder for now, will be filled later)
        """
        self.llm = llm

        # Build the graph
        build_validator = StateGraph(ValidationState)

        build_validator.add_node("validate", self.validate_node)

        build_validator.set_entry_point("validate")
        build_validator.add_edge("validate", END)

        self.validation_agent = build_validator.compile()

    async def validate_node(self, state: ValidationState):
        """
        Node that validates the retrieved clinical trial content.

        Takes the task and retrieved_content as inputs and provides validation.
        Outputs the final retrieved content in the same schema: List[Dict]
        """
        # Load prompt from YAML
        async with aiofiles.open("./prompts/llm_prompt.yaml", "r") as f:
            content = await f.read()
            prompts = yaml.safe_load(content)
            VALIDATION_PROMPT = prompts.get("VALIDATION_PROMPT", "")

        # Prepare context with retrieved content
        retrieved_content_str = "\n\n".join(
            [
                f"Trial {i+1}:\nContent: {item.get('content', '')}\nMetadata: {item.get('metadata', {})}"
                for i, item in enumerate(state.get("retrieved_content", []))
            ]
        )

        messages = [
            SystemMessage(content=VALIDATION_PROMPT),
            HumanMessage(
                content=f"Task: {state.get('task', '')}\n\nRetrieved Content:\n{retrieved_content_str}"
            ),
        ]

        # Use structured output to ensure clean validated content
        validation_result = await self.llm.with_structured_output(
            ValidationOutput
        ).ainvoke(messages)

        return {
            "node_name": "validate",
            "validated_content": validation_result.validated_content,
            "retrieved_content": state.get("retrieved_content", []),
            "task": state.get("task", ""),
        }
