import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
import os
import yaml
import aiofiles

from agents.states import EmailState, EmailOutput

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmailAgent:
    """
    Generates personalized emails for clinical trial patient outreach.
    """

    def __init__(self, llm):
        """
        Initialize the email agent.

        Args:
            llm: Language model instance (placeholder for now, will be filled later)
        """
        self.llm = llm

        # Build the graph
        build_email = StateGraph(EmailState)

        build_email.add_node("generate_email", self.generate_email_node)

        build_email.set_entry_point("generate_email")
        build_email.add_edge("generate_email", END)

        self.email_agent = build_email.compile()

    async def generate_email_node(self, state: EmailState):
        """
        Node that generates a personalized email for a patient.

        Takes the first patient from retrieved_content and generates an email.
        Uses structured output to ensure proper email format.
        """
        # Load prompt from YAML
        async with aiofiles.open("./prompts/llm_prompt.yaml", "r") as f:
            content = await f.read()
            prompts = yaml.safe_load(content)
            EMAIL_PROMPT = prompts.get("EMAIL_PROMPT", "")

        # Get patient data
        patient_data = state.get("patient_data", {})

        # Prepare context
        patient_str = f"Patient Data:\nContent: {patient_data.get('content', '')}\nMetadata: {patient_data.get('metadata', {})}"

        messages = [
            SystemMessage(content=EMAIL_PROMPT),
            HumanMessage(content=f"Task: {state.get('task', '')}\n\n{patient_str}"),
        ]

        # Use structured output to get email subject and body
        email_result = await self.llm.with_structured_output(EmailOutput).ainvoke(
            messages
        )

        return {
            "node_name": "generate_email",
            "email_subject": email_result.email_subject,
            "email_body": email_result.email_body,
            "patient_data": patient_data,
            "task": state.get("task", ""),
        }
