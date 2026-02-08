import asyncio
import logging
import uuid
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite
from dotenv import load_dotenv

from agents.states import _initialize_state, MainState
from agents.retriever_agent import RetrieverAgent
from agents.validation_agent import ValidationAgent
from agents.reranker_agent import RerankerAgent
from agents.explaination_agent import ExplanationAgent
from agents.email_agent import EmailAgent
from agents.telephony_agent import run_telephony_agent

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================== Clinical Trial Agent System =====================


class ClinicalTrialAgent(StateGraph):
    """
    Main Clinical Trial Agent Orchestrator
    Factory pattern implementation to handle async agents
    while maintaining resources at scale
    """

    def __init__(
        self,
        llm,
        retriever_agent,
        validation_agent,
        reranker_agent,
        explanation_agent,
        email_agent,
        conn,
        compiled_graph,
    ):
        self.llm = llm
        self.retriever_agent = retriever_agent
        self.validation_agent = validation_agent
        self.reranker_agent = reranker_agent
        self.explanation_agent = explanation_agent
        self.email_agent = email_agent
        self.conn = conn
        self.clinical_trial_agent = compiled_graph

    @classmethod
    async def build(cls):
        """Build and compile the clinical trial agent system."""
        # Initialize LLM with Gemini
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Node definitions
        async def retriever_node(state: MainState):
            """Retrieve relevant clinical trials."""
            retriever_state = state["retriever_state"]
            retriever_state["task"] = state.get("task", "")
            output = await retriever_agent.ainvoke(retriever_state)

            return {
                "node_name": "retriever",
                "retriever_state": output,
                "retrieved_content": output.get("retrieved_content", []),
            }

        async def validation_node(state: MainState):
            """Validate retrieved clinical trial data."""
            validation_state = state["validation_state"]
            validation_state["task"] = state.get("task", "")
            validation_state["retrieved_content"] = state.get("retrieved_content", [])
            output = await validation_agent.ainvoke(validation_state)

            return {
                "node_name": "validation",
                "validation_state": output,
                "validated_content": output.get("validated_content", []),
            }

        async def reranker_node(state: MainState):
            """Rerank clinical trials by severity/priority."""
            reranker_state = state["reranker_state"]
            reranker_state["task"] = state.get("task", "")
            reranker_state["retrieved_content"] = state.get("validated_content", [])
            output = await reranker_agent.ainvoke(reranker_state)

            return {
                "node_name": "reranker",
                "reranker_state": output,
                "reranked_content": output.get("reranked_content", []),
            }

        async def explanation_node(state: MainState):
            """Generate explanations for ranked clinical trials."""
            explanation_state = state["explanation_state"]
            explanation_state["task"] = state.get("task", "")
            explanation_state["retrieved_content"] = state.get("reranked_content", [])
            output = await explanation_agent.ainvoke(explanation_state)

            return {
                "node_name": "explanation",
                "explanation_state": output,
                "explanations": output.get("explanations", []),
            }

        async def hitl_node(state: MainState):
            """Human in the loop - show first 5 explanations and ask for approval."""
            explanations = state.get("explanations", [])
            first_one = explanations[:1]

            # Format explanations for display
            formatted_explanations = "\n\n".join(
                [f"Patient {i+1}:\n{exp}" for i, exp in enumerate(first_one)]
            )

            # Interrupt for human input
            human_response = interrupt(
                {
                    "query": f"Here is the top patient:\n\n{formatted_explanations}\n\nDo you want to proceed with this patient? (yes/no)"
                }
            )

            return {
                "node_name": "hitl",
                "hitl": human_response,
            }

        async def email_node(state: MainState):
            """Generate email for the first patient."""
            email_state = state["email_state"]
            email_state["task"] = state.get("task", "")

            # Get first patient from reranked content
            reranked_content = state.get("reranked_content", [])
            if reranked_content:
                email_state["patient_data"] = reranked_content[0]

            output = await email_agent.ainvoke(email_state)

            # Update history
            new_history = state.get("history", [])
            new_history.append({"role": "user", "content": state["task"]})
            new_history.append(
                {
                    "role": "assistant",
                    "content": f"Subject: {output.get('email_subject', '')}\n\n{output.get('email_body', '')}",
                }
            )

            return {
                "node_name": "email",
                "email_state": output,
                "email_subject": output.get("email_subject", ""),
                "email_body": output.get("email_body", ""),
                "history": new_history,
            }

        def telephony_fallback():
            """Fallback function for telephony (to be implemented later)."""
            pass

        # Initialize sub-agents
        retriever_agent = RetrieverAgent(llm).retriever_agent
        validation_agent = ValidationAgent(llm).validation_agent
        reranker_agent = RerankerAgent(llm).reranker_agent
        explanation_agent = ExplanationAgent(llm).explanation_agent
        email_agent = EmailAgent(llm).email_agent

        # Build the main graph
        build_clinical_trial = StateGraph(MainState)

        # Add nodes
        build_clinical_trial.add_node("retriever", retriever_node)
        build_clinical_trial.add_node("validation", validation_node)
        build_clinical_trial.add_node("reranker", reranker_node)
        build_clinical_trial.add_node("explanation", explanation_node)
        build_clinical_trial.add_node("hitl", hitl_node)
        build_clinical_trial.add_node("email", email_node)

        # Set up the flow
        build_clinical_trial.set_entry_point("retriever")
        build_clinical_trial.add_edge("retriever", "validation")
        build_clinical_trial.add_edge("validation", "reranker")
        build_clinical_trial.add_edge("reranker", "explanation")
        build_clinical_trial.add_edge("explanation", "hitl")
        build_clinical_trial.add_edge("hitl", "email")
        build_clinical_trial.add_edge("email", END)

        # Set up checkpointer for memory
        conn = await aiosqlite.connect(
            "checkpoints/clinical_trials_checkpoints.sqlite", check_same_thread=False
        )
        memory = AsyncSqliteSaver(conn)
        compile_kwargs = {"checkpointer": memory}

        compiled_graph = build_clinical_trial.compile(**compile_kwargs)

        return cls(
            llm,
            retriever_agent,
            validation_agent,
            reranker_agent,
            explanation_agent,
            email_agent,
            conn,
            compiled_graph,
        )

    async def close(self):
        """Close database connection."""
        if hasattr(self, "conn"):
            await self.conn.close()


class Runner:
    """
    Runner class for the Clinical Trial Agent System.
    Handles both new threads and existing threads for memory continuity.
    """

    def __init__(self, agent: ClinicalTrialAgent):
        self.agent = agent.clinical_trial_agent
        self.threads = []
        self.thread_id = None
        self.config = {}

    async def new_thread(self, Input: str):
        """
        Start a new thread with a fresh clinical trial query.

        Args:
            Input: User's clinical trial search query

        Returns:
            Dictionary with status, thread_id, and output/query
        """
        self.thread_id = str(uuid.uuid4())
        self.config = {"configurable": {"thread_id": self.thread_id}}
        state = _initialize_state(Input)

        result = await self.agent.ainvoke(state, self.config)

        # Check state to see if we are paused/interrupted
        snapshot = await self.agent.aget_state(self.config)
        if snapshot.next:
            # We are paused (likely at HITL)
            return {
                "status": "paused",
                "thread_id": self.thread_id,
                "query": "Waiting for human approval",
                # Explanations might be in the state result passed back or in the snapshot
                # result is the state at interruption.
                "explanations": result.get("explanations", [])[:1],
            }

        return {"status": "completed", "thread_id": self.thread_id, "output": result}

    async def existing_thread(self, Input: str):
        """
        Continue an existing thread (e.g., after HITL response).

        Args:
            Input: User's response (e.g., "yes" for approval)

        Returns:
            Dictionary with status, thread_id, and output
        """
        if not self.thread_id:
            raise ValueError("No existing thread_id to resume")

        snapshot = await self.agent.aget_state(self.config)
        state = dict(snapshot.values)

        # Update with human response
        state["hitl"] = Input

        command = Command(resume={"hitl": Input})
        result = await self.agent.ainvoke(command, config=self.config)

        # Check if we are still paused (unlikely if we just resumed, unless multi-turn HITL)
        # But for this graph, resuming HITL should lead to email and end.
        snapshot = await self.agent.aget_state(self.config)
        if snapshot.next:
            return {
                "status": "paused",
                "thread_id": self.thread_id,
                "query": "Human input requested again",  # Should not happen in this simple graph
            }

        return {
            "status": "completed",
            "thread_id": self.thread_id,
            "output": result,
        }

    async def get_current_state(self, thread_id: str):
        """Get the current state of a specific thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return await self.agent.aget_state(config)


# Test the system
if __name__ == "__main__":

    async def main():
        """Test the clinical trial agent system."""
        user_query = "Find clinical trials for diabetes patients aged 40-60"

        agent = await ClinicalTrialAgent.build()
        runner = Runner(agent)

        print("Starting new thread...")
        response = await runner.new_thread(user_query)
        print("Agent paused for human input:")
        print(response)

        # Simulate doctor approval
        doctor_response = "yes"
        response = await runner.existing_thread(doctor_response)
        print("\nAgent continued execution:")
        print(response)

        await agent.close()

    asyncio.run(main())
