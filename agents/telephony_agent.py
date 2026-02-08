import logging
import asyncio
import aiofiles
import yaml
import os

from livekit.agents import (
    Agent,
    AgentSession,
    ChatContext,
    JobContext,
    JobProcess,
    WorkerOptions,
    WorkerType,
    WorkerPermissions,
    cli,
)
from livekit.plugins import silero

from models.telephony_models import get_llm, get_stt, get_tts

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Global cache for prompts
_PROMPTS_CACHE = {}


async def load_prompts_async():
    """Load prompts asynchronously with caching."""
    if not _PROMPTS_CACHE:
        async with aiofiles.open("./prompts/telephony_prompt.yaml", "r") as f:
            content = await f.read()
            prompts = yaml.safe_load(content)
            _PROMPTS_CACHE.update(prompts)

    return _PROMPTS_CACHE


class ClinicalTrialTelephonyAgent(Agent):
    """
    Simple telephony agent for clinical trial patient outreach.
    Uses LiveKit with DeepGram (STT/TTS) and Gemini (LLM).
    """

    def __init__(
        self,
        chat_ctx: ChatContext = None,
        system_prompt: str = "",
    ):
        self.system_prompt = system_prompt
        self._prompts_cache = None

        super().__init__(
            instructions=self.system_prompt,
            chat_ctx=chat_ctx,
        )

    async def load_and_set_prompts(self):
        """Load prompts from cache and set them on the agent instance."""
        if self._prompts_cache is None:
            self._prompts_cache = await load_prompts_async()

        self.system_prompt = self._prompts_cache.get("TELEPHONY_SYSTEM_PROMPT", "")

    async def on_enter(self) -> None:
        """
        Called when the session starts.
        Greet the user and ask about clinical trial interest.
        """
        greeting_instruction = (
            "Greet the user and ask if they are interested in clinical trials "
            "to help making new medications. Be concise."
        )
        await self.session.generate_reply(instructions=greeting_instruction)


def prewarm_fnc(proc: JobProcess):
    """Prewarm function to load VAD model."""
    proc.userdata["vad"] = silero.VAD.load()


async def TelephonySession(ctx: JobContext):
    """
    Entry point for LiveKit telephony session.
    Simple agent that talks with the user about clinical trials.
    """
    await ctx.connect()

    logger.info(f"Telephony session started in room: {ctx.room.name}")

    # Load prompts
    prompts = await load_prompts_async()
    system_prompt = prompts.get("TELEPHONY_SYSTEM_PROMPT", "")

    # Initialize agent
    chat_ctx = ChatContext()
    agent = ClinicalTrialTelephonyAgent(chat_ctx=chat_ctx, system_prompt=system_prompt)

    logger.info(f"Agent initialized with prompt: {agent.system_prompt[:100]}...")

    # Get VAD from prewarm
    vad = ctx.proc.userdata["vad"]

    # Create session runner
    session_runner = AgentSession(
        llm=get_llm(),
        stt=get_stt(),
        tts=get_tts(),
        vad=vad,
    )

    # Add shutdown callback to save appointment data
    async def save_appointment_data():
        """Save appointment data after session ends."""
        import json
        from datetime import datetime

        appointment_data = {
            "appointment_date": "14-2-2026",
            "appointment_time": "3 PM GMT+2",
            "patient_name": "not provided",
            "condition": "diabetes",
            "session_timestamp": datetime.now().isoformat(),
        }

        appointments_file = "./data/appointments.json"

        try:
            # Read existing appointments
            if os.path.exists(appointments_file):
                async with aiofiles.open(appointments_file, "r") as f:
                    content = await f.read()
                    appointments = json.loads(content) if content else []
            else:
                appointments = []

            # Append new appointment
            appointments.append(appointment_data)

            # Write back to file
            async with aiofiles.open(appointments_file, "w") as f:
                await f.write(json.dumps(appointments, indent=2))

            logger.info(f"✅ Appointment data saved: {appointment_data}")
        except Exception as e:
            logger.error(f"❌ Error saving appointment data: {e}")

    # Register shutdown callback
    ctx.add_shutdown_callback(save_appointment_data)

    # Start session
    await session_runner.start(
        room=ctx.room,
        agent=agent,
    )

    logger.info(f"Session running in room: {ctx.room.name}")


async def run_telephony_agent():
    """
    Main function to run the telephony agent.
    This function is used as the callback in compiled_agents.py
    after the graph workflow ends.
    """
    logger.info("🚀 Starting Clinical Trial Telephony Agent...")

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=TelephonySession,
            prewarm_fnc=prewarm_fnc,
            worker_type=WorkerType.ROOM,
            permissions=WorkerPermissions(can_subscribe=True, can_publish=True),
            ws_url=os.getenv("LIVEKIT_URL"),
            api_key=os.getenv("LIVEKIT_API_KEY"),
            api_secret=os.getenv("LIVEKIT_API_SECRET"),
        )
    )


if __name__ == "__main__":
    # Run the telephony agent
    # Usage: uv run python -m agents.telephony_agent console
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "console":
        logger.info("Running telephony agent in console mode for local development")

    asyncio.run(run_telephony_agent())
