import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import yaml
import aiofiles

from agents.states import ExplanationState, ExplanationOutput

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationAgent:
    """
    Generates explanations for each retrieved clinical trial based on user query.
    """

    def __init__(self, llm):
        """
        Initialize the explanation agent.

        Args:
            llm: Language model instance (placeholder for now, will be filled later)
        """
        self.llm = llm

        # Build the graph
        build_explainer = StateGraph(ExplanationState)

        build_explainer.add_node("explain", self.explain_node)

        build_explainer.set_entry_point("explain")
        build_explainer.add_edge("explain", END)

        self.explanation_agent = build_explainer.compile()

    async def explain_node(self, state: ExplanationState):
        """
        Node that generates explanations for retrieved clinical trials.

        Takes the task (user input) and retrieved_content from validation agent.
        Outputs a List[str] with one explanation for each patient/trial.
        """
        # Load prompt from YAML
        async with aiofiles.open("./prompts/llm_prompt.yaml", "r") as f:
            content = await f.read()
            prompts = yaml.safe_load(content)
            EXPLANATION_PROMPT = prompts.get("EXPLANATION_PROMPT", "")

        # Prepare context with retrieved content
        retrieved_content_str = "\n\n".join(
            [
                f"Trial {i+1}:\nContent: {item.get('content', '')}\nMetadata: {item.get('metadata', {})}"
                for i, item in enumerate(state.get("retrieved_content", []))
            ]
        )

        messages = [
            SystemMessage(content=EXPLANATION_PROMPT),
            HumanMessage(
                content=f"Task: {state.get('task', '')}\n\nRetrieved Content:\n{retrieved_content_str}"
            ),
        ]

        # Use structured output to ensure we get one explanation per trial
        explanation_result = await self.llm.with_structured_output(
            ExplanationOutput
        ).ainvoke(messages)

        return {
            "node_name": "explain",
            "explanations": explanation_result.explanations,
            "retrieved_content": state.get("retrieved_content", []),
            "task": state.get("task", ""),
        }
