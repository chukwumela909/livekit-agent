import os
import logging

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
    JobProcess,
    Agent,
    AgentSession,
)
from livekit.agents.voice.room_io import RoomIO
from livekit.plugins import deepgram, cartesia, openai, silero

logger = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO)


def prewarm(proc: JobProcess):
    """Preload heavy models (VAD) before accepting jobs."""
    logger.info("Preloading Silero VAD...")
    proc.userdata["vad"] = silero.VAD.load()


async def request_fnc(req: JobRequest):
    """Accept any job request."""
    logger.info(f"Accepting job for room {req.room.name}")
    await req.accept()


async def entrypoint(ctx: JobContext):
    """Called when the agent is assigned a job."""
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

    # 4. Configure the conversational agent
    agent = Agent(
        instructions="You are a helpful, friendly voice assistant. Keep responses concise and conversational.",
        stt=deepgram.STT(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            model="nova-2",
            language="en-US",
        ),
        llm=openai.LLM(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
            model="llama-3.1-8b-instant",
            temperature=0.7,
        ),
        tts=cartesia.TTS(
            api_key=os.environ["CARTESIA_API_KEY"],
            voice="a0e99841-438c-4a64-b679-ae501e7d6091",
            model="sonic",
            language="en",
        ),
        vad=vad,
    )

    # 5. Create session and bridge to the LiveKit room
    session = AgentSession(agent=agent)
    room_io = RoomIO(session, ctx.room, participant=participant)

    logger.info("Agent is now running and listening.")
    await room_io.run()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
            prewarm_fnc=prewarm,
        )
    )
