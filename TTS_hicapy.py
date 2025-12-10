import asyncio
import os
import logging
import json
from collections import deque
from typing import Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from dotenv import load_dotenv

# --- Config ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Hicapy")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# --- The Brain: Bot Session State ---
class BotSession:
    def __init__(self, bot_id: str):
        self.bot_id = bot_id
        self.history: List[dict] = []
        
        # QUEUE 1: Text Sentences waiting to be turned into Audio
        self.text_queue: deque = deque()
        
        # QUEUE 2: Audio Chunks (bytes) ready to be sent to client
        self.outbox_audio_queue: asyncio.Queue = asyncio.Queue()
        
        # BUFFER: Holds sentences that were "cut off" to be resumed later
        self.resume_buffer: deque = deque()
        
        self.is_processing = False
        self.interruption_signal = asyncio.Event()

    def clear_audio_outbox(self):
        """Clears audio that hasn't been sent yet."""
        while not self.outbox_audio_queue.empty():
            try:
                self.outbox_audio_queue.get_nowait()
                self.outbox_audio_queue.task_done()
            except asyncio.QueueEmpty:
                break

manager: Dict[str, BotSession] = {}

# --- Core Logic Functions ---

async def tts_worker(session: BotSession):
    """
    Constantly watches text_queue. 
    Converts Text -> Audio -> Puts in Outbox.
    """
    while True:
        try:
            # 1. Wait for text to speak
            if not session.text_queue:
                await asyncio.sleep(0.1)
                continue

            # 2. Pop the next sentence (LIFO or FIFO depending on need, usually FIFO)
            text_chunk = session.text_queue.popleft()
            
            # 3. Check for Interruption before expensive TTS call
            if session.interruption_signal.is_set():
                # If we are interrupted, we DO NOT process this chunk yet.
                # We put it back into the Resume Buffer to speak later.
                session.resume_buffer.appendleft(text_chunk)
                await asyncio.sleep(0.1)
                continue

            # 4. Generate Audio (Simulating PlayAI/TTS)
            # logger.info(f"Bot {session.bot_id} Generating TTS for: {text_chunk[:10]}...")
            
            # Actual TTS Call
            response = await asyncio.to_thread(
                client.audio.speech.create,
                model="playai-tts",
                voice="Fritz-PlayAI",
                input=text_chunk,
                response_format="wav"
            )
            
            # 5. Put in Outbox (if not interrupted during generation)
            if not session.interruption_signal.is_set():
                audio_bytes = response.read()
                await session.outbox_audio_queue.put(response.read())
            else:
                # If interrupted DURING generation, discard audio, save text to resume buffer
                session.resume_buffer.appendleft(text_chunk)

        except Exception as e:
            logger.error(f"TTS Error: {e}")
            await asyncio.sleep(1)

async def llm_stream_worker(session: BotSession, prompt: str):
    """
    Generates text from LLM and pushes sentences to text_queue.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep answers concise."},
        *session.history,
        {"role": "user", "content": prompt}
    ]

    current_sentence = ""
    full_response = ""
    
    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="llama-3.1-8b-instant",
            messages=messages,
            stream=True
        )

        for chunk in completion:
            # Check Interruption - Stop generating completely if interrupted
            if session.interruption_signal.is_set():
                logger.info("LLM generation stopped due to interruption.")
                break

            token = chunk.choices[0].delta.content
            if token:
                full_response += token
                current_sentence += token
                
                # Split by sentence to allow granular interruption
                if token in [".", "?", "!"]:
                    if current_sentence.strip():
                        # Push sentence to queue for TTS
                        session.text_queue.append(current_sentence.strip())
                        current_sentence = ""

        # Handle leftover text
        if current_sentence.strip() and not session.interruption_signal.is_set():
            session.text_queue.append(current_sentence.strip())

        # Save history
        session.history.append({"role": "assistant", "content": full_response})

    except Exception as e:
        logger.error(f"LLM Error: {e}")

# --- WebSocket Endpoint ---

@app.websocket("/ws/{bot_id}")
async def websocket_endpoint(websocket: WebSocket, bot_id: str):
    await websocket.accept()
    
    # Create Session
    session = BotSession(bot_id)
    manager[bot_id] = session
    
    # Start TTS background worker for this bot
    tts_task = asyncio.create_task(tts_worker(session))

    # Background task to push Audio to Client
    async def audio_sender():
        try:
            while True:
                chunk = await session.outbox_audio_queue.get()
                await websocket.send_bytes(chunk)
                session.outbox_audio_queue.task_done()
        except Exception:
            pass

    sender_task = asyncio.create_task(audio_sender())

    try:
        while True:
            # Wait for data (Text or Control Signals)
            data = await websocket.receive_text()

            # === HANDLING INTERRUPTION ===
            if data == "__INTERRUPT__":
                logger.info(f"Bot {bot_id}: Interrupted!")
                
                # 1. Raise Flag (Stops LLM and TTS workers)
                session.interruption_signal.set()
                
                # 2. Clear Audio Outbox (Remove what hasn't been sent yet)
                session.clear_audio_outbox()
                
                # 3. Move pending Text Queue to Resume Buffer
                # (Preserves chunks 4 & 5)
                while session.text_queue:
                    item = session.text_queue.pop() # Pop from right (end)
                    session.resume_buffer.appendleft(item) # Store in order
                
                # 4. Tell Client to flush their buffer
                await websocket.send_text("__CLEAR_LOCAL_BUFFER__")
                continue

            # === HANDLING NEW INPUT (The Doubt) ===
            # Example input: "Wait, explain that part about latency again."
            
            logger.info(f"Received Query: {data}")
            
            # 1. Reset Interruption Flag (Allow processing again)
            session.interruption_signal.clear()
            
            # 2. Run LLM for the Doubt
            # This pushes the answer to the text_queue FIRST
            await llm_stream_worker(session, data)
            
            # 3. Logic: Resume the OLD Queue?
            # User requirement: "answers the doubt first then resumes to the queue whatever is left"
            if session.resume_buffer:
                logger.info(f"Resuming {len(session.resume_buffer)} chunks...")
                
                # Inject a bridge phrase (Optional)
                session.text_queue.append("Now, returning to what I was saying.")
                
                # Move everything from Resume Buffer back to Text Queue
                while session.resume_buffer:
                    session.text_queue.append(session.resume_buffer.popleft())

    except WebSocketDisconnect:
        logger.info(f"Bot {bot_id} disconnected")
        tts_task.cancel()
        sender_task.cancel()
        del manager[bot_id]