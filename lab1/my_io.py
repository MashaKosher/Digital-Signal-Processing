from __future__ import annotations

import wave
from array import array
from typing import Iterable


def write_wav_mono(
    path: str,
    samples: Iterable[float],
    sample_rate_hz: int = 44100,
) -> None:
    """Сохраняет моно WAV (16-bit PCM) из списка/итератора значений [-1, 1]."""
    # Преобразуем во временный массив int16 с насыщением
    ints = array(
        "h",
        (
            _float_to_int16(x)
            for x in samples
        ),
    )

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate_hz)
        wf.writeframes(ints.tobytes())


def _float_to_int16(x: float) -> int:
    # Клиппинг к [-1.0, 1.0]
    if x < -1.0:
        x = -1.0
    elif x > 1.0:
        x = 1.0
    # Масштабируем к диапазону int16
    scaled = int(round(x * 32767.0))
    if scaled < -32768:
        return -32768
    if scaled > 32767:
        return 32767
    return scaled


