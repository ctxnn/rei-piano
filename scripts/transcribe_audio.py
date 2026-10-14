"""Transcribe a song's piano part from an audio file (how we got Runaway's notes).

Run in the transcription environment (separate from training; it needs PyTorch):

    uv venv -p 3.11 .venv-transcribe
    VIRTUAL_ENV=.venv-transcribe uv pip install demucs piano_transcription_inference \
        pretty_midi librosa soundfile "numpy<2"
    .venv-transcribe/bin/python scripts/transcribe_audio.py data/audio/runaway_full.mp3 \
        --seconds 75 --period 1.3793 --first-onset 0.27

Steps (see docs/02_song_transcription.md for why each matters):
  1. Demucs htdemucs_6s splits the mix into stems, including a piano stem.
  2. A piano transcription model turns the piano stem into MIDI notes.
  3. (optional, with --period) For every strike on a regular grid, report which pitch
     gained the most energy at the onset *in the full mix*. The separated stem can lose
     the piano when drums enter, so the mix is the ground truth for verification.

The first ByteDance-model run downloads a 165 MB checkpoint to
~/piano_transcription_inference_data/ (from zenodo.org/record/4034264).
"""

import argparse
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import pretty_midi

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "transcription"


def separate(wav: Path) -> Path:
    subprocess.run(
        [sys.executable, "-m", "demucs", "-n", "htdemucs_6s", "-o", str(OUT / "separation"), str(wav)],
        check=True,
    )  # fmt: skip
    return OUT / "separation" / "htdemucs_6s" / wav.stem / "piano.wav"


def transcribe(piano_wav: Path) -> Path:
    from piano_transcription_inference import PianoTranscription, sample_rate

    audio, _ = librosa.load(piano_wav, sr=sample_rate, mono=True)
    midi_out = OUT / "piano_stem_transcribed.mid"
    PianoTranscription(device="cpu").transcribe(audio, str(midi_out))
    return midi_out


def grid_check(mix_wav: Path, first_onset: float, period: float, candidates: list[str]) -> None:
    """For each strike time, which candidate pitch's energy jumps the most?"""
    y, sr = librosa.load(mix_wav, sr=44100, mono=True)
    freqs = np.fft.rfftfreq(1 << 16, 1 / sr)

    def spectrum(a: int, b: int) -> np.ndarray:
        seg = y[a:b]
        return np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=1 << 16))

    i = 0
    while (t := first_onset + i * period) + 0.5 < len(y) / sr:
        a, b = int((t + 0.04) * sr), int((t + 0.45) * sr)
        pre = int((t - 0.45) * sr)
        after, before = spectrum(a, b), spectrum(pre, pre + (b - a))

        def onset_gain(note: str) -> float:
            hz = librosa.note_to_hz(note)
            band = (freqs > hz * 0.985) & (freqs < hz * 1.015)
            return after[band].max() / (before[band].max() + 5)

        best = max(candidates, key=onset_gain)
        print(f"strike {i:3d}  t={t:6.2f}s  strongest new pitch: {best:4s} (gain {onset_gain(best):5.1f})")
        i += 1


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("audio", type=Path)
    p.add_argument("--seconds", type=float, default=75.0)
    p.add_argument("--period", type=float, help="seconds between strikes, for the grid check")
    p.add_argument("--first-onset", type=float, default=0.0)
    p.add_argument("--candidates", default="C#5,D#5,E5,G#5,A5,B5,C#6,D#6,E6,F#6")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    clip = OUT / f"{args.audio.stem}_first{int(args.seconds)}s.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(args.audio), "-t",
                    str(args.seconds), "-ar", "44100", str(clip)], check=True)  # fmt: skip

    midi = transcribe(separate(clip))
    pm = pretty_midi.PrettyMIDI(str(midi))
    for n in sorted((n for i in pm.instruments for n in i.notes), key=lambda n: n.start):
        name = pretty_midi.note_number_to_name(n.pitch)
        print(f"{n.start:7.2f}s  {n.end - n.start:5.2f}s  {name:4s}  vel {n.velocity}")

    if args.period:
        grid_check(clip, args.first_onset, args.period, args.candidates.split(","))


if __name__ == "__main__":
    main()
