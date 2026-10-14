# 02 · Getting the notes: transcribing Runaway's piano from the recording

The robot is graded against a MIDI file, so a wrong note in the MIDI means the robot
gets trained to play a wrong note perfectly. Online sheet music for Runaway disagreed
(keys of E, B, and A♭ were all claimed; tempos 80–160 BPM), so we measured the
recording itself.

**Result:** [`src/runaway/song.py`](../src/runaway/song.py) holds the notes as data;
[`scripts/make_midi.py`](../scripts/make_midi.py) writes `data/midi/runaway_45s.mid`
(33 strikes) and `data/midi/runaway_60s.mid` (44 strikes).

## The answer

One note at a time, struck once every half note at **87 BPM** (every 1.379 s). An
8-bar, 16-strike phrase repeats unchanged under the drums, bass and vocals:

```
bar:   1        2        3          4          5          6          7        8
      E6  E6   E6  E5   D#6  D#6   D#6  D#5   C#6  C#6   C#6  C#5   A5  A5   G#5  E6
```

* Range C#5–E6 (20 cm of keyboard): a single right hand can play all of it.
* In E major these are scale degrees 1, 7, 6, 4, 3: a descending line where each step is
  hammered three times, then dropped an octave on the 4th strike.
* First strike at 0.27 s in the audio file. Drums and bass enter at ~22 s, vocals ~45 s.

## Method (reproducible)

All steps ran locally on the Mac in `.venv-transcribe` (Python 3.11, `demucs`,
`piano_transcription_inference`, `librosa`). Input: `data/audio/runaway_full.mp3`
(a personal copy, not committed), first 75 s.

1. **Source separation** with Demucs `htdemucs_6s` (it has a dedicated *piano* stem):
   `python -m demucs -n htdemucs_6s -o outputs/separation runaway_first75s.wav`
2. **Automatic transcription** of the piano stem with ByteDance's high-resolution piano
   transcription model (Kong et al. 2021, note F1 0.97 on MAESTRO).
3. **Grid fit.** The 16 loud onsets in the solo intro are spaced 1.38 s apart with
   < 10 ms jitter → half notes at 87.0 BPM (31 intervals span 42.76 s → 86.99 BPM).
4. **Octave check.** The transcriber also reported notes an octave higher (E7 over E6
   etc.). The spectrum shows the upper octave at only ~5% of the fundamental's
   amplitude — a normal piano overtone, not a second key. → single notes.
5. **Fixing a separation artifact.** In the piano stem, strikes at 23–25 s and 43–51 s
   looked silent. Measuring the expected pitch's onset energy in the *full mix* showed
   they are there: Demucs had moved the piano into its "other" stem once the drums
   started. Checking against the full mix is essential.
6. **Verification of all strikes.** For each of the first 53 grid positions, we
   compared the expected pitch's energy just after vs. just before the strike in the
   full mix. 52/53 matched the phrase pattern; the 1 mismatch (52.7 s) is a vocal
   overlapping the note, and the transcriber had confirmed D#6 there.
7. **Alignment test.** Our MIDI, synthesized and overlaid on the record
   (`outputs/audio_check/runaway_60s_overlay.wav`, left ear original, right ear ours):
   44 onsets, mean timing error 3 ms, max 23 ms (one analysis frame), no drift.

## Design choices in the MIDI

* **Gate 0.8.** Each note is held 1.10 s of its 1.38 s slot. Without a gap, three
  consecutive E6 strikes would merge into one long "hold E6" goal and the robot would
  never have to re-strike. The 0.28 s gap is ~5 control steps at 20 Hz.
* **No sustain pedal.** The recording rings, but the pedal is an extra thing to learn
  and isn't needed to be recognisable. Can be added later.
* **Velocities** (accented first strike per bar) are only for nicer reference audio:
  RoboPianist's reward ignores loudness.
* **Time 0 = first strike.** The environment adds 1 s of silence before it
  (`EnvConfig.initial_buffer_time`) so the hand can travel from its start position to E6.

## Ideas not used (yet)

* A left-hand bass part (E–B/D#–C#–A, the song's I–V6–vi–IV) would make a nicer
  two-handed arrangement, but it is not in the record's piano part.
* Basic Pitch (Spotify) as a second transcriber, to cross-check.
