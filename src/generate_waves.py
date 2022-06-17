#!/usr/bin/env python
"""
RUn this script to populate the `audio` directory
"""
import math
from pathlib import Path
from typing import Callable
from typing import Union

import torch
from torchaudio import save  # type: ignore

OUTPUT_PATH = Path(__file__).parent.parent / "audio"
SAMPLE_RATE = 44100
AUDIO_SECS = 1
BIT_DEPTH = 24
FREQ = 440


def saw_wave(t: Union[int, float]) -> float:
    return (t % (2 * math.pi)) / math.pi - 1


def make_waveform(
    frequency: Union[int, float],
    func: Callable,
) -> None:
    time_steps = torch.linspace(0, AUDIO_SECS, SAMPLE_RATE * AUDIO_SECS)
    return func(2 * math.pi * frequency * time_steps).view(1, -1)


if __name__ == "__main__":
    save(
        OUTPUT_PATH / "sine.wav",
        make_waveform(FREQ, torch.sin),
        SAMPLE_RATE,
        bits_per_sample=BIT_DEPTH,
    )
    save(
        OUTPUT_PATH / "saw.wav",
        make_waveform(FREQ, saw_wave),
        SAMPLE_RATE,
        bits_per_sample=BIT_DEPTH,
    )
