"""Synthesise static/ambience.wav (a 60 s drone that loops cleanly) from sine waves and noise, so no licensed music."""

import wave
from pathlib import Path

import numpy as np

RATE = 22050
SECONDS = 60
N = RATE * SECONDS
OUT = Path(__file__).resolve().parent.parent / "static" / "ambience.wav"
t = np.arange(N) / RATE
rng = np.random.default_rng(217)


def periodic(f: float) -> float:
    # round to a frequency that fits a whole number of cycles in the loop, so the end meets the start
    return round(f * SECONDS) / SECONDS


def lowpass_noise(cutoff: float, slope: int = 2) -> np.ndarray:
    spec = np.fft.rfft(rng.normal(size=N))
    spec *= 1 / (1 + (np.fft.rfftfreq(N, 1 / RATE) / cutoff) ** slope)
    x = np.fft.irfft(spec, N)
    return x / np.abs(x).max()


def bandpass_noise(centre: float, width: float) -> np.ndarray:
    spec = np.fft.rfft(rng.normal(size=N))
    f = np.fft.rfftfreq(N, 1 / RATE)
    spec *= np.exp(-((f - centre) / width) ** 2)
    x = np.fft.irfft(spec, N)
    return x / np.abs(x).max()


def place(sound: np.ndarray, start: float) -> np.ndarray:
    # drop a short sound into the loop, wrapping past the end back to the start
    out = np.zeros(N)
    idx = (int(start * RATE) + np.arange(len(sound))) % N
    np.add.at(out, idx, sound)
    return out


def drone() -> np.ndarray:
    # A1 against B-flat1 (a semitone) and E-flat2 (a tritone): the two most uneasy intervals
    voices = [(55.0, 0.5), (55.13, 0.35), (58.27, 0.32), (77.78, 0.2), (110.4, 0.06), (155.6, 0.05)]
    out = sum(a * np.sin(2 * np.pi * periodic(f) * t + rng.uniform(0, 6.28)) for f, a in voices)
    breathe = 0.65 + 0.35 * np.sin(2 * np.pi * periodic(1 / 15) * t) * np.sin(2 * np.pi * periodic(1 / 20) * t + 1)
    return out * breathe


def creak(length: float, f0: float) -> np.ndarray:
    # bowed metal: a few inharmonic partials that bend slightly in pitch
    k = np.arange(int(length * RATE)) / RATE
    env = np.sin(np.pi * k / length) ** 2
    bend = 1 + 0.03 * np.sin(2 * np.pi * k / length * 1.5)
    phase = 2 * np.pi * f0 * np.cumsum(bend) / RATE
    tone = sum(a * np.sin(m * phase) for m, a in [(1, 1), (2.76, 0.5), (5.4, 0.25), (8.9, 0.12)])
    return env * tone


def music_box_note(freq: float) -> np.ndarray:
    k = np.arange(int(3.5 * RATE)) / RATE
    wobble = 1 + 0.004 * np.sin(2 * np.pi * 5 * k)          # slightly out of tune, slightly seasick
    phase = 2 * np.pi * freq * np.cumsum(wobble) / RATE
    return np.exp(-k / 0.9) * (np.sin(phase) + 0.35 * np.sin(2.01 * phase) + 0.1 * np.sin(4.2 * phase))


def knock() -> np.ndarray:
    k = np.arange(int(0.35 * RATE)) / RATE
    return np.exp(-k / 0.05) * np.sin(2 * np.pi * 70 * k) + 0.3 * np.exp(-k / 0.01) * rng.normal(size=len(k))


def reverb(x: np.ndarray, seconds: float = 2.4) -> np.ndarray:
    # circular convolution with a decaying noise tail, so the reverb also wraps around the loop
    n = int(seconds * RATE)
    ir = rng.normal(size=n) * np.exp(-np.arange(n) / (RATE * seconds / 5))
    ir[0] = 0
    padded = np.zeros(N)
    padded[:n] = ir / np.abs(ir).sum() * 40
    return np.fft.irfft(np.fft.rfft(x) * np.fft.rfft(padded), N)


def main() -> None:
    mix = 0.55 * drone()
    mix += 0.22 * lowpass_noise(70, 4) * (0.6 + 0.4 * np.sin(2 * np.pi * periodic(1 / 12) * t))   # rumble
    mix += 0.05 * bandpass_noise(900, 250) * (0.5 + 0.5 * np.sin(2 * np.pi * periodic(1 / 30) * t)) ** 4  # air

    effects = np.zeros(N)
    for start, length, f0 in [(9, 4.5, 310), (31, 3.5, 233), (47, 5.0, 370)]:
        effects += 0.16 * place(creak(length, f0), start)
    # a tritone-heavy tune, played too slowly, with gaps
    for start, freq in [(4, 987.8), (5.6, 698.5), (7.9, 659.3), (21, 932.3), (22.7, 659.3), (38, 987.8),
                        (39.4, 1046.5), (41.5, 698.5), (54, 622.3)]:
        effects += 0.07 * place(music_box_note(freq), start)
    for start in (16.0, 16.45, 44.2, 44.62, 45.1):
        effects += 0.35 * place(knock(), start)

    mix += effects + 0.6 * reverb(effects)
    mix = np.tanh(mix * 1.4)                       # soft clip, a little grit
    mix = mix / np.abs(mix).max() * 0.32
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(OUT), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(RATE)
        f.writeframes((mix * 32767).astype(np.int16).tobytes())
    print(f"wrote {OUT.relative_to(OUT.parents[1])}")


if __name__ == "__main__":
    main()
