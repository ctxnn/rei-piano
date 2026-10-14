"""Write the Runaway MIDI files the robot trains on, plus audio to check them by ear.

Outputs (in data/midi/ and outputs/audio_check/):
  runaway_45s.mid, runaway_60s.mid   - the training targets
  runaway_60s_synth.wav              - our MIDI rendered with a piano soundfont
  runaway_60s_overlay.wav            - original recording + our MIDI, mixed and
                                       time-aligned; if the transcription is right
                                       you hear one piano, not two

Usage:
  .venv/bin/python scripts/make_midi.py
"""

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from runaway import song

ROOT = Path(__file__).resolve().parents[1]
MIDI_DIR = ROOT / "data" / "midi"
AUDIO_CHECK_DIR = ROOT / "outputs" / "audio_check"
ORIGINAL_AUDIO = ROOT / "data" / "audio" / "runaway_full.mp3"
SAMPLE_RATE = 44100


def find_soundfont() -> Path:
    import robopianist

    return Path(robopianist.__file__).parent / "soundfonts" / "TimGM6mb.sf2"


def render(midi_path: Path, wav_path: Path) -> np.ndarray:
    subprocess.run(
        ["fluidsynth", "-ni", "-g", "1.0", "-r", str(SAMPLE_RATE), "-F", str(wav_path),
         str(find_soundfont()), str(midi_path)],
        check=True, capture_output=True,
    )  # fmt: skip
    audio, _ = sf.read(wav_path, always_2d=True)
    return audio.mean(axis=1)


def load_original(seconds: float) -> np.ndarray:
    """Decode the original mp3 to mono float32 with ffmpeg."""
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(ORIGINAL_AUDIO), "-t", str(seconds),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        check=True, capture_output=True,
    ).stdout  # fmt: skip
    return np.frombuffer(raw, dtype=np.float32)


def main() -> None:
    for seconds in (45, 60):
        path = song.write_midi(MIDI_DIR / f"runaway_{seconds}s.mid", seconds)
        n = len(song.strikes(seconds))
        print(f"wrote {path.relative_to(ROOT)}  ({n} strikes)")

    AUDIO_CHECK_DIR.mkdir(parents=True, exist_ok=True)
    synth = render(MIDI_DIR / "runaway_60s.mid", AUDIO_CHECK_DIR / "runaway_60s_synth.wav")

    # Overlay: shift our MIDI so its first strike lands where the record's does.
    original = load_original(62.0)
    shift = int(song.FIRST_STRIKE_IN_AUDIO_S * SAMPLE_RATE)
    aligned = np.zeros_like(original)
    n = min(len(synth), len(original) - shift)
    aligned[shift : shift + n] = synth[:n]
    aligned *= np.abs(original).max() / (np.abs(aligned).max() + 1e-9)
    stereo = np.stack([0.6 * original, 0.6 * aligned], axis=1)  # L = record, R = ours
    sf.write(AUDIO_CHECK_DIR / "runaway_60s_overlay.wav", stereo, SAMPLE_RATE)
    print(f"wrote {AUDIO_CHECK_DIR.relative_to(ROOT)}/runaway_60s_overlay.wav "
          "(left ear = original, right ear = our MIDI)")


if __name__ == "__main__":
    main()
