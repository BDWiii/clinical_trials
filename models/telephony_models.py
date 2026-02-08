import os
from dotenv import load_dotenv
from livekit.plugins import deepgram, google

load_dotenv()


def get_llm():
    """Get Gemini LLM for LiveKit."""
    return google.LLM(
        model="models/gemini-2.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_stt():
    """Get DeepGram STT for LiveKit."""
    return deepgram.STT(
        model=os.getenv("DEEPGRAM_STT"),
        sample_rate=int(os.getenv("DEEPGRAM_SAMPLE_RATE")),
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )


def get_tts():
    """Get DeepGram TTS for LiveKit."""
    return deepgram.TTS(
        model=os.getenv("DEEPGRAM_TTS"),
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        encoding="linear16",
    )
