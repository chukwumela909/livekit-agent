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


async def _publish_transcript(ctx: JobContext, role: str, text: str):
    """Send transcript text to all participants via data messages."""
    try:
        payload = f'{{"role":"{role}","text":"{text}"}}'
        await ctx.room.local_participant.send_text(
            payload,
            topic="transcript",
        )
        logger.info(f"Published {role} transcript: {text[:60]}...")
    except Exception as e:
        logger.warning(f"Failed to publish transcript: {e}")


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

    # 4. Build the agent (behavior + pipeline config)
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

    # 5. Create session and wire up transcript publishing
    session = AgentSession()

    import asyncio

    def _on_user_input(ev):
        text = getattr(ev, "text", "") or getattr(ev, "transcript", "")
        if text:
            asyncio.create_task(_publish_transcript(ctx, "user", text))

    def _on_conversation_item(ev):
        item = getattr(ev, "item", None)
        if item and getattr(item, "role", None) == "assistant":
            text = _extract_text(item)
            if text:
                asyncio.create_task(_publish_transcript(ctx, "agent", text))

    session.on("user_input_transcribed", _on_user_input)
    session.on("conversation_item_added", _on_conversation_item)

    # 6. Start the agent session in the room
    logger.info("Agent is now running and listening.")
    await session.start(agent, room=ctx.room)


def _extract_text(item) -> str:
    """Extract plain text from a ChatMessage or similar object."""
    content = getattr(item, "content", None)
    if isinstance(content, list):
        texts = []
        for c in content:
            if isinstance(c, str):
                texts.append(c)
            elif hasattr(c, "text"):
                texts.append(str(c.text))
        return " ".join(texts).strip()
    if isinstance(content, str):
        return content.strip()
    return str(content) if content else ""


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            request_fnc=request_fnc,
            prewarm_fnc=prewarm,
        )
    )
