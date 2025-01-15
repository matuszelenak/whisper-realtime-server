import logging
from typing import List, TypeVar, Tuple, AsyncGenerator

import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

T = TypeVar("T")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.ERROR)

SAMPLING_RATE = 16000


def split_to_batches(array: List[T], batch_size: int) -> List[List[T]]:
    return [array[i:i + batch_size] for i in range(0, len(array), batch_size)]


async def even_chunks(audio_chunks: AsyncGenerator[np.ndarray, None], chunk_size):
    buffer: np.ndarray = np.zeros((chunk_size, ), np.float32)
    i = 0
    async for chunk in audio_chunks:
        if chunk is None:
            break

        for sample in chunk:
            buffer[i] = sample

            i += 1
            if i == chunk_size:
                yield buffer

                i = 0
                buffer = np.zeros((chunk_size, ), np.float32)

    if i > 0:
        yield buffer[:i]

    yield None


async def continuous_transcriber(
        model: WhisperModel,
        audio_chunks: AsyncGenerator[np.ndarray, None],
        stable_words_for_iterations=2,
        stable_words_before_advance=2
) -> AsyncGenerator[dict, None]:
    transcribed_samples = np.array([], dtype=np.float32)

    segments: List[Tuple[List[str], int]] = []

    stable_segment_counter = 0
    current_ts_offset = 0.0
    transcribed_chain: List[str] = []
    sample_offset = 0
    async for chunk in even_chunks(audio_chunks, 4000):
        if chunk is None:
            yield {
                'complete': True,
                'words': transcribed_chain,
                'start': current_ts_offset,
                'id': stable_segment_counter,
                'samples': transcribed_samples.shape[0],
                'final': True
            }

            return

        transcribed_samples = np.concatenate((transcribed_samples, chunk), axis=0)
        if transcribed_samples.shape[0] < (SAMPLING_RATE / 2):
            continue

        t_segments, info = model.transcribe(transcribed_samples, beam_size=5, vad_filter=True)

        segment: Segment

        transcribed_chain = []
        for segment in t_segments:
            if segment.no_speech_prob >= 0.45:
                continue

            transcribed_chain.extend([x for x in segment.text.split(' ') if x != ''])

        if len(transcribed_chain) == 0:
            continue

        # print(transcribed_chain)

        # scipy.io.wavfile.write(f'./transcript/{sample_offset}-{sample_offset + transcribed_samples.shape[0]}__{"_".join(transcribed_chain)}.wav', SAMPLING_RATE, transcribed_samples)

        segments.append((transcribed_chain, transcribed_samples.shape[0]))

        # Find a reference segment, that contains ONLY words, that occur in some segments transcribed later
        last_3_segments_sequences = [' '.join(seg[0]) for seg in segments[-3:]]
        if len(last_3_segments_sequences) == 3 and len(set(last_3_segments_sequences)) == 1:
            best_segment_index = len(segments) - 1
        else:
            best_segment_index, best_segment_length = -1, -1
            for ref_s_i, (ref_segment_words, ref_segment_samples) in enumerate(segments):
                if len(ref_segment_words) < stable_words_before_advance:
                    continue

                ref_sequence = ' '.join([w.lower() for w in ref_segment_words])
                sequence_repetitions = 1

                for com_segment_words, com_segment_samples in segments[ref_s_i + 1:]:
                    comp_sequence = ' '.join([w.lower() for w in com_segment_words])

                    if comp_sequence[:len(ref_sequence)] == ref_sequence:
                        sequence_repetitions += 1

                if sequence_repetitions >= stable_words_for_iterations:
                    if len(ref_segment_words) > best_segment_length:
                        best_segment_length = len(ref_segment_words)
                        best_segment_index = ref_s_i

        if best_segment_index != -1:
            best_s_words, best_s_samples = segments[best_segment_index]

            yield {
                'complete': True,
                'words': best_s_words,
                'start': current_ts_offset,
                'id': stable_segment_counter,
                'samples': transcribed_samples.shape[0]
            }

            # transcribed_samples = np.array([], dtype=np.float32)
            transcribed_samples = transcribed_samples[best_s_samples:]
            segments = segments[best_segment_index + 1:]
            stable_segment_counter += 1
            current_ts_offset += best_s_samples / SAMPLING_RATE
            sample_offset += best_s_samples

        else:
            yield {
                'complete': False,
                'words': transcribed_chain,
                'samples': transcribed_samples.shape[0],
                'start': current_ts_offset,
            }
