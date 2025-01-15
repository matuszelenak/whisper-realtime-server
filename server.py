import base64
import logging
import os
import uuid
from contextlib import asynccontextmanager

import starlette
from fastapi import FastAPI
from faster_whisper import WhisperModel, BatchedInferencePipeline
from starlette.websockets import WebSocket
import numpy as np

from transcriber import continuous_transcriber

transcriber: WhisperModel = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global transcriber

    faster_whisper_custom_model_path = '/mnt/home/WhisperLive/assets/fw-large-v3/'

    if not os.path.exists(faster_whisper_custom_model_path):
        raise ValueError(f"Custom faster_whisper model '{faster_whisper_custom_model_path}' is not a valid path.")
    logging.info("Custom model option was provided. Switching to single model mode.")

    model_size = "large-v3"
    model_path = '/mnt/home/WhisperLive/assets/fw-large-v3'

    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    transcriber = BatchedInferencePipeline(model=model)
    yield

app = FastAPI(lifespan=lifespan)


logger = logging.getLogger(__name__)


@app.websocket('/transcribe')
async def transcribe_ws(websocket: WebSocket):
    global transcriber
    await websocket.accept()

    async def samples_generator():
        while True:
            data = await websocket.receive_json()

            if data.get('commit', False):
                yield None

            else:
                samples = base64.b64decode(data['samples'])
                samples = np.frombuffer(samples, dtype=np.float32)

                yield samples

    try:
        while True:
            async for segment in continuous_transcriber(transcriber, samples_generator()):
                await websocket.send_json(segment)

    except starlette.websockets.WebSocketDisconnect:
        pass

    except Exception as e:
        logger.error('Error in LLM task', exc_info=True)
        logger.error(str(e))
