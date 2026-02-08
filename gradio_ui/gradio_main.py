import gradio as gr
import asyncio
import logging
from typing import List, Dict, Optional, Tuple
import sys
import os
import subprocess

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.compiled_agents import ClinicalTrialAgent, Runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===================== Global State =====================
agent_instance = None
runner_instance = None


# ===================== Filter Options =====================

SEX_OPTIONS = ["ALL", "FEMALE", "MALE", "Not Specified"]

STUDY_TYPE_OPTIONS = ["OBSERVATIONAL", "INTERVENTIONAL", "EXPANDED_ACCESS"]

STUDY_STATUS_OPTIONS = [
    "COMPLETED",
    "TERMINATED",
    "RECRUITING",
    "UNKNOWN",
    "NOT_YET_RECRUITING",
    "WITHDRAWN",
    "ACTIVE_NOT_RECRUITING",
    "SUSPENDED",
    "APPROVED_FOR_MARKETING",
    "ENROLLING_BY_INVITATION",
    "NO_LONGER_AVAILABLE",
    "AVAILABLE",
]

AGE_GROUP_OPTIONS = [
    "ADULT, OLDER_ADULT",
    "ADULT",
    "CHILD, ADULT",
    "CHILD, ADULT, OLDER_ADULT",
    "CHILD",
    "OLDER_ADULT",
]

STUDY_RESULTS_OPTIONS = ["ALL", "YES", "NO"]

# Placeholder for study titles - will be loaded dynamically
STUDY_TITLES = []


# ===================== Helper Functions =====================


async def initialize_agent():
    """Initialize the clinical trial agent system."""
    global agent_instance, runner_instance
    if agent_instance is None:
        logger.info("Initializing Clinical Trial Agent...")
        agent_instance = await ClinicalTrialAgent.build()
        runner_instance = Runner(agent_instance)
        logger.info("Agent initialized successfully!")
    return runner_instance


def load_study_titles():
    """Load study titles from database or file."""
    # TODO: Implement actual loading from database
    # For now, return placeholder
    return ["Study 1", "Study 2", "Study 3"]


def format_filters(
    sex, study_type, study_status, age_group, study_title, study_results
):
    """Format filter selections into a readable string."""
    filters = []

    if sex and sex != "ALL":
        filters.append(f"Sex: {sex}")
    if study_type:
        filters.append(f"Type: {', '.join(study_type)}")
    if study_status:
        filters.append(f"Status: {', '.join(study_status)}")
    if age_group:
        filters.append(f"Age: {', '.join(age_group)}")
    if study_title:
        filters.append(f"Title: {study_title}")
    if study_results and study_results != "ALL":
        filters.append(f"Results: {study_results}")

    return " | ".join(filters) if filters else "No filters applied"


def build_query_with_filters(
    user_query, sex, study_type, study_status, age_group, study_title, study_results
):
    """Build enhanced query with filter criteria."""
    filter_parts = []

    if sex and sex != "ALL":
        filter_parts.append(f"sex: {sex}")
    if study_type:
        filter_parts.append(f"study type: {', '.join(study_type)}")
    if study_status:
        filter_parts.append(f"status: {', '.join(study_status)}")
    if age_group:
        filter_parts.append(f"age group: {', '.join(age_group)}")
    if study_title:
        filter_parts.append(f"specific study: {study_title}")
    if study_results and study_results != "ALL":
        filter_parts.append(f"results available: {study_results}")

    if filter_parts:
        filter_str = " AND ".join(filter_parts)
        enhanced_query = f"{user_query}\n\nFilter criteria: {filter_str}"
        return enhanced_query

    return user_query


# ===================== Main Chat Function =====================


async def chat_interface(
    message: str,
    history: List[Dict[str, str]],
    sex: str,
    study_type: List[str],
    study_status: List[str],
    age_group: List[str],
    study_title: Optional[str],
    study_results: str,
):
    """
    Main chat function that processes user queries and integrates with agents.
    """
    if not message.strip():
        yield history
        return

    try:
        # Initialize agent if needed
        runner = await initialize_agent()

        # Build query with filters
        enhanced_query = build_query_with_filters(
            message,
            sex,
            study_type,
            study_status,
            age_group,
            study_title,
            study_results,
        )

        # Add user message to history
        history.append({"role": "user", "content": message})

        # Show processing indicator
        # Add a temporary assistant message for loading state
        history.append(
            {"role": "assistant", "content": "🔄 Processing your request..."}
        )
        yield history

        # Call the agent
        logger.info(f"Sending query to agent: {enhanced_query}")

        # Start new thread or continue existing
        if runner.thread_id is None:
            response = await runner.new_thread(enhanced_query)
        else:
            response = await runner.existing_thread(enhanced_query)

        # Format response
        if response.get("status") == "paused":
            # HITL - show explanations
            explanations = response.get("explanations", [])
            formatted_response = "**Top Candidate:**\n\n"
            for i, exp in enumerate(explanations, 1):
                formatted_response += f"**Patient {i}:**\n{exp}\n\n"
            formatted_response += "\n\n*Do you want to proceed with these patients? (Reply 'yes' or 'no')*"
        elif response.get("status") == "completed":
            output = response.get("output", {})
            email_subject = output.get("email_subject", "")
            email_body = output.get("email_body", "")

            if email_subject and email_body:
                formatted_response = f"Subject: {email_subject}\n\n{email_body}"
            else:
                formatted_response = "✅ Request completed successfully!"
        else:
            formatted_response = "⚠️ Unexpected response from agent."

        # Update history with response (replace the loading message)
        history[-1] = {"role": "assistant", "content": formatted_response}
        yield history

    except Exception as e:
        logger.error(f"Error in chat interface: {e}", exc_info=True)
        # Update or add error message
        if history and history[-1]["role"] == "assistant":
            history[-1] = {"role": "assistant", "content": f"❌ Error: {str(e)}"}
        else:
            history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        yield history


def apply_filters(sex, study_type, study_status, age_group, study_title, study_results):
    """Apply filters and return status message."""
    filter_summary = format_filters(
        sex, study_type, study_status, age_group, study_title, study_results
    )
    # TODO: Actually query database to get count
    return f"🔍 **Active Filters:**\n{filter_summary}\n\n*Filters will be applied to your next query.*"


def reset_filters():
    """Reset all filters to default values."""
    return (
        "ALL",  # sex
        [],  # study_type
        ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING"],  # study_status
        [],  # age_group
        None,  # study_title
        "ALL",  # study_results
        "✅ All filters have been reset to defaults.",
    )


def end_session_and_call():
    """End session and launch telephony agent in console."""
    try:
        # Launch telephony agent in a new process
        # We use Popen to run it without blocking the UI
        # Ensure we run from the project root and inherit environment
        # This allows the agent to use the same terminal for input/output
        subprocess.Popen(
            ["uv", "run", "python", "-m", "agents.telephony_agent", "console"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=os.environ.copy(),
        )
        return [], "📞 Telephony Agent Launched! Check your terminal."
    except Exception as e:
        logger.error(f"Error launching telephony agent: {e}")
        return [], f"❌ Error launching agent: {e}"


# ===================== Gradio Interface =====================


def create_interface():
    """Create the main Gradio interface."""

    # Custom CSS for medical-grade styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Roboto', sans-serif !important;
    }
    .chat-container {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .filter-panel {
        background-color: #F5F5F5;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .primary-btn {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    .secondary-btn {
        background-color: #00897B !important;
        color: white !important;
    }
    """

    with gr.Blocks(
        css=custom_css, title="Clinical Trial Patient Selection System"
    ) as interface:

        gr.Markdown(
            """
            # 🏥 Clinical Trial Patient Selection System
            ### AI-Powered Patient Matching and Outreach Platform
            """
        )

        with gr.Row():
            # Left Column: Chat Interface (75% width)
            with gr.Column(scale=3, elem_classes="chat-container"):
                gr.Markdown("### 💬 Chat Interface")

                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=500,
                    show_label=False,
                    avatar_images=(None, "🤖"),
                )

                with gr.Row():
                    msg = gr.Textbox(
                        label="Your Message",
                        placeholder="Ask about clinical trials, patient eligibility, or request specific information...",
                        lines=2,
                        max_lines=5,
                        scale=4,
                    )
                    send_btn = gr.Button("Send 📤", variant="primary", scale=1)

                # Example queries
                gr.Examples(
                    examples=[
                        "Find patients with diabetes eligible for the new cardiac trial",
                        "Top patients with heart failure for recruitment",
                        "Identify patients aged 40-60 with liver disease for recruitment",
                        "List patients with high severity scores for immediate review",
                    ],
                    inputs=msg,
                    label="💡 Example Queries",
                )

                clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary", size="sm")
                end_btn = gr.Button(
                    "End Session & Call Agent 📞", variant="stop", size="sm"
                )

            # Right Column: Filtration Panel (25% width)
            with gr.Column(scale=1, elem_classes="filter-panel"):
                gr.Markdown("### 🔍 Filter Options")

                sex_filter = gr.Dropdown(
                    label="Sex", choices=SEX_OPTIONS, value="ALL", interactive=True
                )

                study_type_filter = gr.Dropdown(
                    label="Study Type",
                    choices=STUDY_TYPE_OPTIONS,
                    multiselect=True,
                    interactive=True,
                )

                study_status_filter = gr.Dropdown(
                    label="Study Status",
                    choices=STUDY_STATUS_OPTIONS,
                    value=["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING"],
                    multiselect=True,
                    interactive=True,
                )

                age_group_filter = gr.Dropdown(
                    label="Age Group",
                    choices=AGE_GROUP_OPTIONS,
                    multiselect=True,
                    interactive=True,
                )

                study_title_filter = gr.Dropdown(
                    label="Study Title",
                    choices=load_study_titles(),
                    interactive=True,
                    allow_custom_value=True,
                )

                study_results_filter = gr.Dropdown(
                    label="Study Results Available",
                    choices=STUDY_RESULTS_OPTIONS,
                    value="ALL",
                    interactive=True,
                )

                gr.Markdown("---")

                with gr.Row():
                    apply_btn = gr.Button(
                        "Apply Filters ✓", variant="primary", size="sm"
                    )
                    reset_btn = gr.Button("Reset ↺", variant="secondary", size="sm")

                filter_status = gr.Markdown("*No filters applied*")

        # Event handlers
        filter_inputs = [
            sex_filter,
            study_type_filter,
            study_status_filter,
            age_group_filter,
            study_title_filter,
            study_results_filter,
        ]

        # Chat submission
        msg.submit(
            chat_interface, inputs=[msg, chatbot] + filter_inputs, outputs=chatbot
        ).then(lambda: "", outputs=msg)

        send_btn.click(
            chat_interface, inputs=[msg, chatbot] + filter_inputs, outputs=chatbot
        ).then(lambda: "", outputs=msg)

        # Clear chat
        clear_btn.click(lambda: [], outputs=chatbot)

        # Apply filters
        apply_btn.click(apply_filters, inputs=filter_inputs, outputs=filter_status)

        # Reset filters
        reset_btn.click(reset_filters, outputs=filter_inputs + [filter_status])

        # End session and call agent
        end_btn.click(end_session_and_call, outputs=[chatbot, msg])

        gr.Markdown(
            """
            ---
            **Note:** This system integrates multiple AI agents for patient selection, validation, ranking, and outreach.
            All patient data is handled securely and in compliance with HIPAA regulations.
            """
        )

    return interface


# ===================== Main Entry Point =====================

if __name__ == "__main__":
    interface = create_interface()
    interface.queue()  # Enable queuing for async operations
    interface.launch(
        server_name="0.0.0.0", server_port=7860, share=False, show_error=True
    )
