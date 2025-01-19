import base64
import logging
from contextlib import asynccontextmanager

import numpy as np
import starlette
from fastapi import FastAPI
from faster_whisper import WhisperModel, BatchedInferencePipeline
from starlette.websockets import WebSocket

from transcriber import continuous_transcriber

transcriber: WhisperModel = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global transcriber

    model_size = "distil-large-v3"

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
                if len(segment['words']) > 0:
                    await websocket.send_json(segment)

    except starlette.websockets.WebSocketDisconnect:
        pass

    except Exception as e:
        logger.error('Exception occured', exc_info=True)
        logger.error(str(e))
