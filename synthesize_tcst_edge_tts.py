#!/usr/bin/env python3
"""
Batch synthesize Chinese text from TCST text.json using edge-tts.

Features:
- Parse JSON entries: sample_id -> text.Chinese
- Async synthesis with concurrency control
- Retry on transient failures
- Progress bar (tqdm)
- Save as WAV (16kHz, mono, 16-bit PCM)

Dependencies:
  pip install edge-tts pydub tqdm

Note:
- pydub requires ffmpeg installed on your system.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import edge_tts
from pydub import AudioSegment
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize Chinese speech from text.json with edge-tts."
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=Path("TCST/text.json"),
        help="Path to input JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/TCST/wav_zh"),
        help="Directory to save output WAV files.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="zh-CN-XiaoxiaoNeural",
        help="edge-tts voice name.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=20,
        help="Max number of concurrent synthesis tasks.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Retry times for failed synthesis.",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=1.5,
        help="Base wait seconds before retry (uses linear backoff).",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        default=Path("data/TCST/wav_zh_errors.log"),
        help="Path to error log file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output wav files.",
    )
    return parser.parse_args()


def load_items(json_path: Path) -> List[Tuple[str, str]]:
    with json_path.open("r", encoding="utf-8") as f:
        data: Dict[str, dict] = json.load(f)

    items: List[Tuple[str, str]] = []
    for sample_id, payload in data.items():
        zh_text = payload.get("text", {}).get("Chinese", "")
        if isinstance(zh_text, str):
            zh_text = zh_text.strip()
        else:
            zh_text = ""
        items.append((sample_id, zh_text))
    return items


def convert_mp3_bytes_to_wav_16k_mono(audio_bytes: bytes, out_wav_path: Path) -> None:
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    audio.export(out_wav_path, format="wav")


async def synthesize_one(
    sample_id: str,
    text: str,
    out_wav_path: Path,
    voice: str,
    retries: int,
    retry_wait: float,
) -> Tuple[bool, str]:
    if not text:
        return False, "Empty Chinese text"

    for attempt in range(1, retries + 1):
        try:
            communicate = edge_tts.Communicate(text=text, voice=voice)
            audio_buf = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buf.extend(chunk["data"])

            if not audio_buf:
                raise RuntimeError("No audio bytes received from edge-tts")

            await asyncio.to_thread(
                convert_mp3_bytes_to_wav_16k_mono, bytes(audio_buf), out_wav_path
            )
            return True, ""
        except Exception as exc:
            if attempt < retries:
                await asyncio.sleep(retry_wait * attempt)
            else:
                return False, f"{type(exc).__name__}: {exc}"

    return False, "Unknown error"


async def run(args: argparse.Namespace) -> None:
    items = load_items(args.json_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.error_log.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    total = len(items)
    done = 0
    ok_count = 0
    skip_count = 0
    fail_count = 0

    pbar = tqdm(total=total, desc="Synthesizing", unit="utt")

    async def process_one(sample_id: str, text: str) -> None:
        nonlocal done, ok_count, skip_count, fail_count

        out_wav_path = args.out_dir / f"{sample_id}.wav"
        if out_wav_path.exists() and not args.overwrite:
            async with lock:
                skip_count += 1
                done += 1
                pbar.update(1)
            return

        async with sem:
            success, msg = await synthesize_one(
                sample_id=sample_id,
                text=text,
                out_wav_path=out_wav_path,
                voice=args.voice,
                retries=args.retries,
                retry_wait=args.retry_wait,
            )

        async with lock:
            if success:
                ok_count += 1
            else:
                fail_count += 1
                with args.error_log.open("a", encoding="utf-8") as f:
                    f.write(f"{sample_id}\t{msg}\n")
            done += 1
            pbar.update(1)

    tasks = [asyncio.create_task(process_one(sample_id, text)) for sample_id, text in items]
    await asyncio.gather(*tasks)
    pbar.close()

    print(f"Total     : {total}")
    print(f"Succeeded : {ok_count}")
    print(f"Skipped   : {skip_count}")
    print(f"Failed    : {fail_count}")
    print(f"Output dir: {args.out_dir}")
    print(f"Error log : {args.error_log}")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

