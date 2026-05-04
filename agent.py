import os
import logging

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    JobProcess,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import deepgram, cartesia, openai, silero

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)


async def prewarm(proc: JobProcess):
    """Preload heavy models (VAD) before accepting jobs.

    This runs once when the worker process starts, not per-room.
    """
    logger.info("Preloading Silero VAD...")
    proc.userdata["vad"] = silero.VAD.load()


async def request_fnc(req: JobRequest):
    """Accept any job request (any user joining any room triggers the agent)."""
    logger.info(f"Accepting job for room {req.room.name}")
    await req.accept(entrypoint)


async def entrypoint(ctx: JobContext):
    """Called when the agent is assigned a job (someone joined a room)."""
    room_name = ctx.room.name
    logger.info(f"Starting agent for room: {room_name}")

    # 1. Connect to the room and auto-subscribe to audio tracks
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # 2. Wait for at least one human participant to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # 3. Retrieve preloaded VAD from worker userdata
    vad = ctx.proc.userdata.get("vad")
    if vad is None:
        logger.warning("VAD not preloaded; falling back to on-demand load.")
        vad = silero.VAD.load()

    # 4. Configure the conversational pipeline
    # Groq is OpenAI-compatible; we use the OpenAI plugin with a custom base_url.
    assistant = VoicePipelineAgent(
        vad=vad,
        stt=deepgram.STT(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            model="nova-2",               # streaming STT optimized for conversation
            language="en-US",
        ),
        llm=openai.LLM(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",   # lowest-latency Groq model
            temperature=0.7,
        ),
        tts=cartesia.TTS(
            api_key=os.environ["CARTESIA_API_KEY"],
            voice="a0e99841-438c-4a64-b679-ae501e7d6091",  # warm conversational default
            model="sonic",                   # ultra-low latency
            language="en",
        ),
    )

    # 5. Start the assistant — it will automatically listen to participant audio,
    #    run the STT -> LLM -> TTS pipeline, and publish synthesized audio back.
    assistant.start(ctx.room, participant)
    logger.info("Agent is now running and listening.")

    # Optional: send a brief audible confirmation when ready
    # (The agent will speak automatically once the user stops talking.)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
            prewarm_fnc=prewarm,
        )
    )
