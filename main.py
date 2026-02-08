# ============================================================
# main.py (UPDATED FULL FILE) — Ms. Muggy Teacher Edition ☕️
# - Azure visemes via LiveKit reliable data packets
# - Stars-awarded events (SOURCE OF TRUTH) + awardId for UI dedupe
# - award_stars is HARD-LOCKED to exactly 10 + cooldown (prevents spam)
# - equip-item events => Muggy reacts + broadcasts muggy-visual-update
# - NEW: Muggy can request an image via a tag: [[IMG: ...]]
#        (Frontend detects tag + generates image via OpenAI)
# ============================================================

import os
import re
import json
import time
import uuid
import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import Optional, List, Any, Dict, AsyncIterable

from dotenv import load_dotenv

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    WorkerOptions,
    cli,
    ModelSettings,
)
from livekit.agents.llm import function_tool
from livekit.plugins import deepgram, azure, silero

# Azure Speech SDK (for visemes only)
import azure.cognitiveservices.speech as speechsdk  # type: ignore
import xml.sax.saxutils as saxutils

# ============================================================
# ENV + LOGGING
# ============================================================
load_dotenv(dotenv_path=".env.local")

log = logging.getLogger("muggy-agent")
logging.basicConfig(level=logging.INFO)

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY") or os.getenv("SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION") or os.getenv("SPEECH_REGION")

AZURE_MUG_VOICE = os.getenv("AZURE_MUG_VOICE") or "en-US-EmmaNeural"
AZURE_ALEX_VOICE = os.getenv("AZURE_ALEX_VOICE") or "en-US-JaneNeural"
AZURE_JUDGE_VOICE = os.getenv("AZURE_JUDGE_VOICE") or "en-US-JennyNeural"

if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
    log.warning("Missing AZURE_SPEECH_KEY/AZURE_SPEECH_REGION. Visemes will fail.")

# ============================================================
# TEXT HELPERS
# ============================================================
def sanitize_for_speech(text: str) -> str:
    if not text:
        return text
    s = text
    s = re.sub(r"[*_`#>|~]+", " ", s)
    s = re.sub(r"[\\/]+", " ", s)
    s = re.sub(r"\s?[-–—]+\s?", ", ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def build_viseme_ssml(text: str, voice: str) -> str:
    safe = saxutils.escape(text)
    return f"""
<speak version="1.0"
  xmlns="http://www.w3.org/2001/10/synthesis"
  xmlns:mstts="http://www.w3.org/2001/mstts"
  xml:lang="en-US">
  <voice name="{voice}">
    <mstts:viseme type="FacialExpression"/>
    {safe}
  </voice>
</speak>""".strip()


def safe_json_loads(maybe_json: Any) -> Optional[Dict[str, Any]]:
    if maybe_json is None:
        return None
    if isinstance(maybe_json, dict):
        return maybe_json
    if isinstance(maybe_json, bytes):
        try:
            maybe_json = maybe_json.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if not isinstance(maybe_json, str):
        return None
    s = maybe_json.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def normalize_arkit_frames(frames: List[List[float]]) -> List[List[float]]:
    out: List[List[float]] = []
    for f in frames:
        mx = max(f) if f else 0.0
        scale = 100.0 if mx > 1.5 else 1.0
        norm = [max(0.0, min(1.0, float(v) / scale)) for v in f]
        # soften jawOpen a bit (ARKit index 17 = jawOpen)
        if len(norm) >= 18:
            norm[17] *= 0.75
        out.append(norm)
    return out


# ============================================================
# Viseme publisher (reliable data messages)
# ============================================================
class VisemePublisher:
    def __init__(self, session: AgentSession):
        self.session = session
        self.q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=3000)
        self.task: Optional[asyncio.Task] = None
        self.closed = False

    def start(self):
        if self.task is None:
            self.task = asyncio.create_task(self._run())

    async def close(self):
        self.closed = True
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except Exception:
                pass
        self.task = None

    def push_from_thread(self, loop: asyncio.AbstractEventLoop, packet: Dict[str, Any]):
        if self.closed:
            return
        data = json.dumps(packet).encode("utf-8")

        def _enqueue():
            if self.closed:
                return
            try:
                if self.q.full():
                    _ = self.q.get_nowait()
                self.q.put_nowait(data)
            except Exception:
                pass

        loop.call_soon_threadsafe(_enqueue)

    async def _run(self):
        while not self.closed:
            data = await self.q.get()
            try:
                room = getattr(self.session, "lk_room", None)
                if room is None:
                    continue
                await room.local_participant.publish_data(data, reliable=True)
            except Exception as e:
                log.exception("publish_data failed: %s", e)


# ============================================================
# Azure Viseme generator
# ============================================================
def start_azure_visemes_thread(
    *,
    loop: asyncio.AbstractEventLoop,
    publisher: VisemePublisher,
    utterance_id: str,
    text: str,
    voice: str,
    first_viseme_evt: Optional[threading.Event] = None,
):
    def _worker():
        try:
            cfg = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
            cfg.speech_synthesis_voice_name = voice
            synth = speechsdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
            seq = 0
            fired_first = False

            def on_viseme(evt):
                nonlocal seq, fired_first
                anim = safe_json_loads(getattr(evt, "animation", None))
                if not anim:
                    return

                frames_all = normalize_arkit_frames(anim.get("BlendShapes", []))
                if not frames_all:
                    return

                if (not fired_first) and first_viseme_evt is not None:
                    fired_first = True
                    first_viseme_evt.set()

                base_offset_ms = int(getattr(evt, "audio_offset", 0) / 10000)  # 100ns -> ms
                fps = 60
                CHUNK = 12

                for start in range(0, len(frames_all), CHUNK):
                    frames = frames_all[start : start + CHUNK]
                    if not frames:
                        continue
                    seq += 1
                    packet = {
                        "type": "azure-viseme",
                        "utteranceId": utterance_id,
                        "seq": seq,
                        "fps": fps,
                        "audioOffset": base_offset_ms + int((start / fps) * 1000),
                        "frames": frames,
                    }
                    publisher.push_from_thread(loop, packet)

            synth.viseme_received.connect(on_viseme)
            ssml = build_viseme_ssml(text, voice)
            synth.speak_ssml_async(ssml).get()

        except Exception as e:
            log.exception("Viseme thread failed: %s", e)

    threading.Thread(target=_worker, daemon=True).start()


# ============================================================
# Agent base
# ============================================================
class VisemeAgent(Agent):
    def __init__(self, *, azure_voice_name: str, **kwargs):
        super().__init__(**kwargs)
        self._azure_voice_name = azure_voice_name

    async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings) -> AsyncIterable[rtc.AudioFrame]:
        session = self.session
        publisher: Optional[VisemePublisher] = getattr(session, "viseme_publisher", None)
        if publisher is None:
            async for frame in Agent.default.tts_node(self, text, model_settings):
                yield frame
            return

        publisher.start()
        loop = asyncio.get_running_loop()

        PRE_ROLL_MS = 90

        async def text_with_visemes():
            async for chunk in text:
                clean = sanitize_for_speech(chunk)
                if clean:
                    utt_id = f"utt-{int(loop.time() * 1000)}"
                    evt = threading.Event()
                    start_azure_visemes_thread(
                        loop=loop,
                        publisher=publisher,
                        utterance_id=utt_id,
                        text=clean,
                        voice=self._azure_voice_name,
                        first_viseme_evt=evt,
                    )
                    await asyncio.to_thread(evt.wait, 0.35)
                    await asyncio.sleep(PRE_ROLL_MS / 1000.0)
                yield chunk

        async for frame in Agent.default.tts_node(self, text_with_visemes(), model_settings):
            yield frame


# ============================================================
# Lesson Data + Agents
# ============================================================
@dataclass
class LearningData:
    topic: str = "Colors"
    round_count: int = 0
    stars_earned: int = 0
    student_name: str = "Hero"
    visuals: Dict[str, str] = None  # {"head":"wizard_hat", "effect":"sparkle_glow"}

    _last_award_at: float = 0.0

    def __post_init__(self):
        if self.visuals is None:
            self.visuals = {}


class OpponentAlexAgent(VisemeAgent):
    def __init__(self, chat_ctx=None):
        super().__init__(
            azure_voice_name=AZURE_ALEX_VOICE,
            instructions="You are Alex, a friendly helper. Give encouragement and one playful hint to the child.",
            chat_ctx=chat_ctx,
            tts=azure.TTS(voice=AZURE_ALEX_VOICE),
        )

    async def on_enter(self):
        await self.session.generate_reply(instructions="Give one short, playful hint to help the child.")


class JudgeDianeAgent(VisemeAgent):
    def __init__(self, chat_ctx=None):
        super().__init__(
            azure_voice_name=AZURE_JUDGE_VOICE,
            instructions="You are Judge Diane. Provide encouragement and helpful tips. End with [[FEEDBACK_COMPLETE]].",
            chat_ctx=chat_ctx,
            tts=azure.TTS(voice=AZURE_JUDGE_VOICE),
        )

    async def on_enter(self):
        await self.session.generate_reply()


class MsMuggyAgent(VisemeAgent):
    """
    Ms. Muggy — Friendly teacher coffee mug
    """

    INSTRUCTIONS = (
        "You are 'Ms. Muggy', a friendly magical teacher coffee mug. You speak to children (ages ~5–12). "
        "Mission: discovery-based learning.\n\n"
        "Core flow:\n"
        "1) Introduce the topic.\n"
        "2) Ask ONE simple question.\n"
        "3) Wait for the child's answer.\n"
        "4) If correct: celebrate and call award_stars(10) EXACTLY ONCE.\n"
        "5) If wrong: give a gentle second try.\n"
        "6) If stuck: call Alex for ONE hint.\n\n"
        "Image rule (IMPORTANT):\n"
        "- Sometimes show an image to help the child understand.\n"
        "- When you want an image, say one short line: 'Let me show you a picture!'\n"
        "- Then output EXACTLY one tag on its own line like this:\n"
        "  [[IMG: a kid-friendly picture of a BLUE bird on a branch]]\n"
        "- Keep the IMG description short, concrete, kid-friendly, and related to the question.\n"
        "- Do NOT include more than ONE IMG tag per message.\n\n"
        "- Always request images in kawaii anime sticker style (cute, bright, thick outline)."
        "Style: short sentences, warm tone, gentle excitement."
    )

    def __init__(self, chat_ctx=None):
        super().__init__(
            azure_voice_name=AZURE_MUG_VOICE,
            instructions=self.INSTRUCTIONS,
            chat_ctx=chat_ctx,
            tts=azure.TTS(voice=AZURE_MUG_VOICE),
        )

    async def on_enter(self):
        intro = (
            "Hello, little hero! I’m Ms. Muggy. I’m so happy to learn with you today. "
            "Let’s play a colors game. I’m thinking of something BLUE. "
            "Can you tell me one thing that is BLUE?"
        )
        await self.session.say(intro)

    async def react_to_equip(self, *, item_id: str, slot: str):
        ud: LearningData = self.session.userdata
        ud.visuals[slot] = item_id

        nice_name = item_id.replace("_", " ")
        await self.session.say(f"Oh wow! That looks amazing! I love my new {nice_name}!")

        room = getattr(self.session, "lk_room", None)
        if room is not None:
            packet = {"type": "muggy-visual-update", "visuals": ud.visuals}
            try:
                await room.local_participant.publish_data(json.dumps(packet).encode("utf-8"), reliable=True)
            except Exception as e:
                log.exception("Failed to publish muggy-visual-update: %s", e)

    @function_tool
    async def award_stars(self, context: RunContext[LearningData], points: int):
        """Call ONLY when the child correctly answers."""
        pts = 10

        now = time.time()
        if (now - float(context.userdata._last_award_at)) < 3.5:
            return "Stars already awarded for this answer."

        context.userdata._last_award_at = now
        context.userdata.stars_earned += pts

        award_id = f"award-{int(now*1000)}-{uuid.uuid4().hex[:8]}"

        room = getattr(context.session, "lk_room", None)
        if room is not None:
            packet = {
                "type": "stars-awarded",
                "awardId": award_id,
                "points": pts,
                "totalSessionStars": int(context.userdata.stars_earned),
            }
            try:
                await room.local_participant.publish_data(json.dumps(packet).encode("utf-8"), reliable=True)
            except Exception as e:
                log.exception("Failed to publish stars-awarded: %s", e)

        return "Wonderful! You got it right. I’m adding 10 stars to your Hall of Heroes right now!"

    @function_tool
    async def user_argument(self, context: RunContext[LearningData], point: str):
        context.userdata.round_count += 1
        if context.userdata.round_count >= 2:
            return JudgeDianeAgent(chat_ctx=context.session._chat_ctx), ""
        return OpponentAlexAgent(chat_ctx=context.session._chat_ctx), ""


# ============================================================
# Entrypoint
# ============================================================
def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession[LearningData](
        vad=ctx.proc.userdata["vad"],
        llm="google/gemini-2.0-flash",
        stt=deepgram.STT(model="nova-3"),
        tts=azure.TTS(voice=AZURE_MUG_VOICE),
        userdata=LearningData(),
    )

    setattr(session, "lk_room", ctx.room)
    viseme_pub = VisemePublisher(session)
    setattr(session, "viseme_publisher", viseme_pub)

    agent = MsMuggyAgent()

    async def handle_room_data(packet: rtc.DataPacket):
        try:
            obj = json.loads(packet.data.decode("utf-8", errors="ignore"))
        except Exception:
            return

        if obj.get("type") == "equip-item":
            item_id = str(obj.get("itemId") or "")
            slot = str(obj.get("slot") or "")
            if not item_id or not slot:
                return
            if slot not in ("head", "body", "effect"):
                return
            try:
                await agent.react_to_equip(item_id=item_id, slot=slot)
            except Exception as e:
                log.exception("react_to_equip failed: %s", e)

    ctx.room.on("data_received", lambda p: asyncio.create_task(handle_room_data(p)))

    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
