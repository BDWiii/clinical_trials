import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import yaml
import aiofiles

from agents.states import RerankerState, RerankedOutput

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RerankerAgent:
    """
    Reranks retrieved clinical trials based on severity and priority.
    """

    def __init__(self, llm):
        """
        Initialize the reranker agent.

        Args:
            llm: Language model instance (placeholder for now, will be filled later)
        """
        self.llm = llm

        # Build the graph
        build_reranker = StateGraph(RerankerState)

        build_reranker.add_node("rerank", self.rerank_node)

        build_reranker.set_entry_point("rerank")
        build_reranker.add_edge("rerank", END)

        self.reranker_agent = build_reranker.compile()

    async def rerank_node(self, state: RerankerState):
        """
        Node that reranks clinical trials by severity and priority.

        Takes retrieved_content as input and provides reranked_content as output.
        Uses structured output to ensure proper List[Dict] format.
        """
        # Load prompt from YAML
        async with aiofiles.open("./prompts/llm_prompt.yaml", "r") as f:
            content = await f.read()
            prompts = yaml.safe_load(content)
            RERANKER_PROMPT = prompts.get("RERANKER_PROMPT", "")

        # Prepare context with retrieved content
        retrieved_content_str = "\n\n".join(
            [
                f"Trial {i+1}:\nContent: {item.get('content', '')}\nMetadata: {item.get('metadata', {})}"
                for i, item in enumerate(state.get("retrieved_content", []))
            ]
        )

        messages = [
            SystemMessage(content=RERANKER_PROMPT),
            HumanMessage(
                content=f"Task: {state.get('task', '')}\n\nRetrieved Content:\n{retrieved_content_str}"
            ),
        ]

        # Use structured output to ensure clean reranked content
        reranked_result = await self.llm.with_structured_output(RerankedOutput).ainvoke(
            messages
        )

        return {
            "node_name": "rerank",
            "reranked_content": reranked_result.reranked_content,
            "retrieved_content": state.get("retrieved_content", []),
            "task": state.get("task", ""),
        }
