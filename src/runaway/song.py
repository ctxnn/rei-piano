"""The piano part of "Runaway" (Kanye West, 2010), as data.

This file is the single source of truth for *what* the robot must play. Everything
here was measured from the recording, not guessed; see docs/02_song_transcription.md
for the evidence (source separation + automatic transcription + per-strike spectral
checks against the full mix).

What the piano does in the first ~70 seconds
--------------------------------------------
One note at a time, struck every half note at 87 BPM (one strike every 1.379 s).
The pattern is an 8-bar phrase of 16 strikes that repeats, unchanged, while drums,
bass and vocals come and go around it:

    bar:   1      2       3       4       5       6       7      8
          E6 E6  E6 E5   D#6 D#6 D#6 D#5  C#6 C#6 C#6 C#5  A5 A5  G#5 E6

In words: three high E's then the E an octave lower, the same shape on D# and on C#,
then A A G# and back to E to start the next phrase. Every note sits between C#5 and
E6, so a single right hand can reach all of it.

Timing conventions
------------------
* Time 0.0 is the first strike. (In the audio file that strike is at 0.27 s.)
* Notes are shortened slightly ("gate" < 1) so that repeated strikes of the same key
  are separate presses. Without a gap the goal would say "keep holding E6" across
  three strikes and the robot would never need to lift its finger, which sounds wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pretty_midi

# Measured from the recording: 31 strike intervals span 42.76 s -> 1.3794 s per
# half note -> 86.99 BPM. We use the round number.
TEMPO_BPM = 87.0
SECONDS_PER_STRIKE = 2 * 60.0 / TEMPO_BPM  # A strike every half note (two beats).

# Offset of the first strike in data/audio/runaway_full.mp3. Only used to line up
# our MIDI with the original recording when making comparison audio / videos.
FIRST_STRIKE_IN_AUDIO_S = 0.27

# One 8-bar phrase = 16 strikes. Note names use scientific pitch (C4 = middle C).
PHRASE: tuple[str, ...] = (
    "E6", "E6", "E6", "E5",
    "D#6", "D#6", "D#6", "D#5",
    "C#6", "C#6", "C#6", "C#5",
    "A5", "A5", "G#5", "E6",
)  # fmt: skip

# The first strike of every bar is played a little harder in the recording, and the
# low note that closes each 2-bar group rings longest. Velocity does not affect the
# robot's reward (RoboPianist only scores whether keys are down), but it makes the
# reference audio sound closer to the record.
_ACCENT_VELOCITY = 100
_NORMAL_VELOCITY = 85


@dataclass(frozen=True)
class Strike:
    index: int  # 0, 1, 2, ... counted from the first strike of the song.
    pitch: str  # e.g. "E6"
    start: float  # seconds
    end: float  # seconds

    @property
    def midi_number(self) -> int:
        return pretty_midi.note_name_to_number(self.pitch)


def strikes(duration_s: float, gate: float = 0.8) -> list[Strike]:
    """All strikes whose onset falls inside the first `duration_s` seconds.

    Args:
        duration_s: How much of the song to include (e.g. 45 or 60).
        gate: Fraction of the 1.379 s slot during which the key is held down.
            0.8 holds for 1.10 s and leaves a 0.28 s (about 5 control steps at
            20 Hz) gap to lift the finger before the next strike.
    """
    if not 0.0 < gate <= 1.0:
        raise ValueError(f"gate must be in (0, 1], got {gate}")
    out = []
    i = 0
    while (start := i * SECONDS_PER_STRIKE) < duration_s:
        out.append(
            Strike(
                index=i,
                pitch=PHRASE[i % len(PHRASE)],
                start=start,
                end=start + gate * SECONDS_PER_STRIKE,
            )
        )
        i += 1
    return out


def to_pretty_midi(duration_s: float, gate: float = 0.8) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=TEMPO_BPM)
    piano = pretty_midi.Instrument(program=0, name="Runaway piano")
    for s in strikes(duration_s, gate):
        accent = s.index % 2 == 0  # First strike of each bar.
        piano.notes.append(
            pretty_midi.Note(
                velocity=_ACCENT_VELOCITY if accent else _NORMAL_VELOCITY,
                pitch=s.midi_number,
                start=s.start,
                end=s.end,
            )
        )
    pm.instruments.append(piano)
    return pm


def write_midi(path: str | Path, duration_s: float, gate: float = 0.8) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pretty_midi(duration_s, gate).write(str(path))
    return path


if __name__ == "__main__":
    for s in strikes(60.0):
        bar = s.index // 2 + 1
        print(f"strike {s.index:2d}  bar {bar:2d}  {s.pitch:>3s}  {s.start:6.2f}s - {s.end:6.2f}s")
