import os
import io
import wave
import json
import logging
import numpy as np
from scipy import signal
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field
from typing import List, Optional
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
# --- Configuration ---
# API Key must be set in environment variable: GROQ_API_KEY
# Models
LLM_MODEL_ID = "llama-3.1-8b-instant" 
TTS_MODEL_ID = "playai-tts"
TTS_VOICE_ID = "Aaliyah-PlayAI"
SAMPLE_RATE_REQUIRED = 16000  # Spec requirement
LOG_LEVEL = logging.INFO

# Setup Logging
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("hicapy")

# Initialize FastAPI
app = FastAPI(title="hicapy/Text Microservice")

# Initialize Groq Client
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
except Exception as e:
    logger.error("Failed to initialize Groq Client. Ensure GROQ_API_KEY is set.")
    raise

# --- Data Models (Pydantic) ---

class GenerateTextRequest(BaseModel):
    bot_id: str
    system_prompt: str
    summary_memory: Optional[str] = ""
    recent_messages: List[str] = []
    user_message: str

class GenerateTextResponse(BaseModel):
    bot_id: str
    response_text: str
    should_speak: bool
    confidence: float = 0.0
    reason: Optional[str] = None

class GenerateAudioRequest(BaseModel):
    bot_id: str
    system_prompt: str
    summary_memory: Optional[str] = ""
    recent_messages: List[str] = []
    user_message: str
    voice: str = "default"
    force_tts: bool = False

# --- Helper Functions ---

class AudioProcessor:
    @staticmethod
    def process_wav_bytes(input_wav_bytes: bytes, target_rate: int = 16000) -> bytes:
        """
        Reads a WAV file from bytes, resamples it to target_rate (16kHz),
        converts to Mono, and returns a valid WAV file as bytes.
        """
        try:
            # 1. Read the input WAV
            with wave.open(io.BytesIO(input_wav_bytes), 'rb') as source_wav:
                # Extract parameters
                original_rate = source_wav.getframerate()
                channels = source_wav.getnchannels()
                width = source_wav.getsampwidth()
                n_frames = source_wav.getnframes()
                
                # Read raw PCM data
                raw_data = source_wav.readframes(n_frames)

            # 2. Convert to numpy array
            # Assuming 16-bit depth (width=2). If 24/32, conversion needed.
            if width == 2:
                dtype = np.int16
            elif width == 1:
                dtype = np.int8
            else:
                # Fallback or strict requirement check. 
                # Most TTS returns 16-bit or 32-bit float.
                # For safety, let's assume int16 for now or fail.
                dtype = np.int16
            
            audio_data = np.frombuffer(raw_data, dtype=dtype)

            # 3. Convert Stereo to Mono if needed
            if channels == 2:
                # Reshape to (frames, 2)
                audio_data = audio_data.reshape(-1, 2)
                # Average channels
                audio_data = audio_data.mean(axis=1).astype(np.int16)
            
            # 4. Resample if needed
            if original_rate != target_rate:
                num_samples = int(len(audio_data) * target_rate / original_rate)
                audio_data = signal.resample(audio_data, num_samples).astype(np.int16)

            # 5. Write to new WAV container
            output_io = io.BytesIO()
            with wave.open(output_io, 'wb') as out_wav:
                out_wav.setnchannels(1)       # Mono
                out_wav.setsampwidth(2)       # 16-bit
                out_wav.setframerate(target_rate)
                out_wav.writeframes(audio_data.tobytes())
            
            return output_io.getvalue()

        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio processing error: {str(e)}")

# --- Endpoints ---

@app.post("/generate-text", response_model=GenerateTextResponse)
async def generate_text(request: GenerateTextRequest):
    """
    Decides what the bot should say and if it should speak.
    """
    try:
        # Construct the messages for the LLM
        messages = [
            {"role": "system", "content": request.system_prompt + "\n\n" + 
             "CRITICAL INSTRUCTION: You MUST return a valid JSON object. " +
             "Do not include markdown formatting like ```json ... ```. " +
             "Format: {\"response_text\": \"...\", \"should_speak\": boolean, \"confidence\": float, \"reason\": \"...\"}"},
            {"role": "user", "content": f"Context Summary: {request.summary_memory}\n" +
                                        f"Recent Chat History: {request.recent_messages}\n" +
                                        f"Current User Input: {request.user_message}"}
        ]

        completion = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=messages,
            temperature=0.7,
            response_format={"type": "json_object"} # Forces JSON output
        )

        content = completion.choices[0].message.content
        logger.info(f"LLM Raw Response: {content}")
        
        # Parse JSON
        decision_data = json.loads(content)

        return GenerateTextResponse(
            bot_id=request.bot_id,
            response_text=decision_data.get("response_text", ""),
            should_speak=decision_data.get("should_speak", False),
            confidence=decision_data.get("confidence", 0.0),
            reason=decision_data.get("reason", "Generated by Groq")
        )

    except json.JSONDecodeError:
        logger.error("Failed to parse LLM JSON response")
        return GenerateTextResponse(
            bot_id=request.bot_id,
            response_text="I encountered an error processing that.",
            should_speak=True,
            confidence=0.0,
            reason="JSON Parse Error"
        )
    except Exception as e:
        logger.error(f"Generate text error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-audio")
async def generate_audio(request: GenerateAudioRequest):
    """
    Generates WAV audio using Groq's TTS (PlayAI) if should_speak criteria are met.
    """
    # 1. Decision Logic (Re-eval or use flag)
    # The spec implies this endpoint receives the TEXT in 'response_text' usually,
    # BUT the request body in spec shows 'user_message' and 'system_prompt', 
    # implying we might need to regenerate the text OR the spec assumes the caller 
    # passes the text they want spoken?
    
    # RE-READING SPEC CAREFULLY: 
    # Request Body has: system_prompt, summary_memory, recent_messages, user_message.
    # It DOES NOT have "text_to_speak".
    # This means this endpoint acts as a "Decision + Generation" or "Regeneration" endpoint.
    # HOWEVER, usually /generate-audio in this pattern implies we assume the decision was "Yes".
    # But to be safe and avoid "Text Drift" (generating different text than /generate-text),
    # ideally, the bot should pass the text it wants spoken.
    # Since the spec doesn't have a 'text' field in the request, we must ASK THE LLM AGAIN what to say,
    # OR assume the 'user_message' IS the script (unlikely).
    
    # INTERPRETATION: The bot hits /generate-text first. If true, it hits /generate-audio.
    # Since /generate-audio has the same context fields, we must re-run the generation 
    # to get the text to speak. This ensures the audio matches the context.
    
    # Logic:
    # 1. If force_tts is False, we check if we should speak (using LLM).
    # 2. If should_speak is False, return 204.
    # 3. If should_speak is True, generate Audio.

    try:
        # Step A: Get the text to speak (and confirm intent)
        text_to_speak = ""
        should_speak = False

        if request.force_tts:
            should_speak = True
            # If forced, we still need text. We ask LLM "What response for this input?"
            # We assume the same prompt logic as /generate-text
            text_response = await generate_text(GenerateTextRequest(
                bot_id=request.bot_id,
                system_prompt=request.system_prompt,
                summary_memory=request.summary_memory,
                recent_messages=request.recent_messages,
                user_message=request.user_message
            ))
            text_to_speak = text_response.response_text
        else:
            # Check logic naturally
            text_response = await generate_text(GenerateTextRequest(
                bot_id=request.bot_id,
                system_prompt=request.system_prompt,
                summary_memory=request.summary_memory,
                recent_messages=request.recent_messages,
                user_message=request.user_message
            ))
            should_speak = text_response.should_speak
            text_to_speak = text_response.response_text

        # Step B: Return 204 if silence is preferred
        if not should_speak and not request.force_tts:
            return Response(status_code=status.HTTP_204_NO_CONTENT)

        # Step C: Generate Audio via Groq TTS
        logger.info(f"Generating audio for: '{text_to_speak[:30]}...'")
        
        tts_response = client.audio.speech.create(
            model=TTS_MODEL_ID,
            voice=TTS_VOICE_ID, # Replace with specific voice ID if needed
            input=text_to_speak,
            response_format="wav" # Groq supports wav
        )

        # tts_response.content contains the binary WAV data (likely 24kHz or 44.1kHz)
        raw_wav_content = tts_response.read()

        # Step D: Process Audio (Resample to 16kHz Mono)
        processed_wav = AudioProcessor.process_wav_bytes(raw_wav_content, SAMPLE_RATE_REQUIRED)

        # Step E: Return Response
        return Response(
            content=processed_wav,
            media_type="audio/wav",
            headers={
                "Content-Length": str(len(processed_wav)),
                "X-Bot-ID": request.bot_id,
                "Content-Disposition": 'attachment; filename="output.wav"'
            }
        )

    except Exception as e:
        logger.error(f"Generate audio error: {e}")
        raise HTTPException(status_code=500, detail=str(e))