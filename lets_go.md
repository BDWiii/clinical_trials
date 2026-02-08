# 🚀 Quick Start Guide - Clinical Trial System

## Running the Clinical Trial Agent System

### 1. **Run the Main Agent Workflow** (compiled_agents.py)

```bash
# Navigate to project directory
cd /Users/mac/Desktop/Projects/clinical_trials

# Run the compiled agent system
uv run python -m agents.compiled_agents
```

This will:
- Initialize all 5 subagents (Retriever → Validation → Reranker → Explanation → Email)
- Process a test query
- Show HITL (Human-in-the-Loop) for doctor approval
- Generate email for selected patient

---

### 2. **Run the Telephony Agent** (telephony_agent.py)

```bash
# Navigate to project directory
cd /Users/mac/Desktop/Projects/clinical_trials

# Run in console mode for local development
uv run python -m agents.telephony_agent console
```

This will:
- Start LiveKit telephony agent
- Use DeepGram for STT/TTS
- Use Gemini for LLM
- Greet user and ask about clinical trial interest
- Save appointment data to `data/appointments.json` after session ends

---

### 3. **Run the Gradio UI** (gradio_main.py)

```bash
# Navigate to project directory
cd /Users/mac/Desktop/Projects/clinical_trials

# Run the Gradio interface
uv run python gradio_ui/gradio_main.py
```

This will:
- Launch Gradio web interface at `http://localhost:7860`
- Provide chat interface with filter panel
- Integrate with all agents
- Handle HITL workflow visually

**Access the UI:**
- Open browser to: `http://localhost:7860`
- Or if using share mode: Check terminal for public URL

---

## Quick Test Commands

### Test Preprocessing Pipeline

```bash
# Preprocess CSV to JSON
uv run python -m preprocessing.save_to_json

# Index data into Chroma vector store
uv run python -m preprocessing.indexing
```

---

## Environment Setup Reminder

Make sure your `.env` file has:

```env
# Gemini (LLM & Embedding)
GOOGLE_API_KEY=your_key_here

# DeepGram (STT & TTS)
DEEPGRAM_API_KEY=your_key_here
DEEPGRAM_STT=nova-2-general
DEEPGRAM_TTS=aura-asteria-en
DEEPGRAM_SAMPLE_RATE=16000
DEEPGRAM_EOT=0.5
DEEPGRAM_EAGER_EOT=0.4
```

---

## Troubleshooting

**If you get import errors:**
```bash
# Install dependencies
uv sync
```

**If Chroma DB is not found:**
```bash
# Make sure you've run the indexing step
uv run python -m preprocessing.indexing
```

**If LiveKit fails:**
- Check that DeepGram API key is valid
- Ensure you're running in console mode for local testing

---

## Notes

- All agents use async/await - make sure to handle properly
- Gradio UI requires the agent to be initialized (may take a few seconds on first run)
- Telephony agent saves appointments automatically on session end
- HITL requires user input - reply with "yes" or "no"
