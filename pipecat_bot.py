import asyncio
import os
from dotenv import load_dotenv
from livekit import api

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import EndFrame, LLMMessagesAppendFrame
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams

load_dotenv()

async def main():
    room_url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    room_name = os.getenv("LIVEKIT_ROOM", "voice-room")

    # Generate bot token
    token = api.AccessToken(
        api_key=api_key, api_secret=api_secret,
    ).with_identity("pipecat-bot").with_name("Pipecat Bot").with_grants(
        api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True)
    )
    token_str = token.to_jwt()

    print(f"[Pipecat] Connecting to {room_url}/{room_name}")

    transport = LiveKitTransport(
        url=room_url,
        token=token_str,
        room_name=room_name,
        params=LiveKitParams(
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            vad_audio_passthrough=True,
        ),
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        settings=DeepgramSTTService.Settings(model="nova-2", language="en-US", punctuate=True),
    )

    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        settings=GroqLLMService.Settings(model="llama-3.1-8b-instant", temperature=0.7),
    )

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        settings=CartesiaTTSService.Settings(
            voice="a0e99841-438c-4a64-b679-ae501e7d6091",
            model="sonic",
        ),
    )

    pipeline = Pipeline([transport.input(), stt, llm, tts, transport.output()])
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant_id):
        print(f"[Pipecat] Participant joined: {participant_id}")
        await task.queue_frames([
            LLMMessagesAppendFrame([
                {"role": "system", "content": "You are a helpful voice assistant. Keep responses concise."}
            ])
        ])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant_id, reason):
        print(f"[Pipecat] Participant left: {participant_id}")
        await task.queue_frame(EndFrame())

    runner = PipelineRunner()
    await runner.run(task)
    print("[Pipecat] Pipeline ended")

if __name__ == "__main__":
    asyncio.run(main())
