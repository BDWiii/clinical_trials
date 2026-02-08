import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import yaml
import aiofiles

from tools.retrieval_tool import clinical_trial_retrieval
from agents.states import RetrieverState, RetrievalQuery

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RetrieverAgent:
    """
    Retrieves relevant clinical trials from the vector store based on user query.
    """

    def __init__(self, llm):
        """
        Initialize the retriever agent.

        Args:
            llm: Language model instance (placeholder for now, will be filled later)
        """
        self.llm = llm
        self.retrieval_function = clinical_trial_retrieval

        # Build the graph
        build_retriever = StateGraph(RetrieverState)

        build_retriever.add_node("retrieve", self.retrieve_node)

        build_retriever.set_entry_point("retrieve")
        build_retriever.add_edge("retrieve", END)

        self.retriever_agent = build_retriever.compile()

    async def retrieve_node(self, state: RetrieverState):
        """
        Node that performs semantic retrieval from the clinical trials vector store.

        Uses structured output to control the LLM's use of the retrieval tool.
        The LLM will generate a query and specify the number of results to retrieve.

        Pydantic Schema (to be implemented):
        - query: str (the search query)
        - k: int (number of patients to retrieve, default=10)
        """
        # Load prompt from YAML
        async with aiofiles.open("./prompts/llm_prompt.yaml", "r") as f:
            content = await f.read()
            prompts = yaml.safe_load(content)
            RETRIEVER_PROMPT = prompts.get("RETRIEVER_PROMPT", "")

        messages = [
            SystemMessage(content=RETRIEVER_PROMPT),
            HumanMessage(content=state.get("task", "")),
        ]

        # Use structured output to get the retrieval query
        # This ensures the LLM provides a properly formatted query
        retrieval_params = await self.llm.with_structured_output(
            RetrievalQuery
        ).ainvoke(messages)

        # Perform the retrieval using the tool
        retrieved_results = await self.retrieval_function.ainvoke(
            {"query": retrieval_params.query, "k": retrieval_params.k}
        )

        return {
            "node_name": "retrieve",
            "retrieved_content": retrieved_results,
            "task": state.get("task", ""),
        }
