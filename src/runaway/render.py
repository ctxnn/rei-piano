"""Turn a recorded performance (.npz from evaluate.py) into a video with sound.

    python -m runaway.render runs/<run>/evals/step_000500000.npz --camera piano/back

How it works:
  * Video: for every recorded control step we put the saved joint positions back into
    MuJoCo (no physics is simulated) and render a frame. 20 control steps per second
    of music -> 20 fps, which we upsample to 40 fps by rendering halfway states.
  * Audio: from the recorded key activations we rebuild exactly what the robot played
    (a note starts when a key goes down and ends when it comes up), write it as MIDI,
    and synthesize it with FluidSynth. So you hear the robot, mistakes included.
  * A caption shows the training step and the F1 score.
  * Loudness is normalised (EBU R128, -16 LUFS, like streaming services), so videos
    are not quiet. An untrained robot that presses nothing gives a silent video: you
    hear exactly what it played.
  * Optional `--backing-track data/audio/runaway_full.mp3`: mixes the original record
    quietly underneath, time-aligned to the first strike, so you hear the robot
    playing along with the song. (Your own copy of the song; keep such videos private.)

Cameras: piano/back, piano/closeup, piano/left, piano/right, piano/topdown,
piano/egocentric.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pretty_midi
import tyro

from runaway.env import EnvConfig, make_env

SAMPLE_RATE = 44100


@dataclass
class RenderArgs:
    recording: Path
    out: Path | None = None  # default: next to the recording, .mp4
    camera: str = "piano/back"
    width: int = 1280
    height: int = 720
    soundfont: Path | None = None  # default: RoboPianist's bundled TimGM6mb.sf2
    caption: bool = True
    # Used only when the run's config.json (runs/<run>/config.json, next to evals/) is
    # missing; otherwise the scene is rebuilt from that file (embodiment, piano height...).
    midi_path: str = "data/midi/runaway_60s.mid"
    embodiment: str = "hands"  # "hands" or "g1"
    # Original recording to mix underneath (optional), and its level relative to the
    # robot's piano.
    backing_track: Path | None = None
    backing_volume: float = 0.35
    # Seconds of silence before the first note in the recording (EnvConfig.initial_buffer_time).
    lead_in: float = 1.0
    # Frames per control step. 2 -> 40 fps (smooth, for showcase videos); 1 -> 20 fps
    # (half the render time, used for the dashboard's many videos).
    substeps: int = 2
    # H.264 quality (lower = better, bigger). 18 is visually lossless (~22 MB for 60 s of
    # 720p); the dashboard uses ~30 for small files (~1 MB at 480x270).
    crf: int = 18
    audio_kbps: int = 160


def activations_to_midi(key_activation: np.ndarray, dt: float) -> pretty_midi.PrettyMIDI:
    """(T, 88) booleans -> MIDI notes: one note per contiguous run of 'key down'."""
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    padded = np.vstack([np.zeros((1, 88), bool), key_activation.astype(bool), np.zeros((1, 88), bool)])
    changes = np.diff(padded.astype(np.int8), axis=0)
    for key in range(88):
        starts = np.flatnonzero(changes[:, key] == 1)
        ends = np.flatnonzero(changes[:, key] == -1)
        for s, e in zip(starts, ends):
            inst.notes.append(pretty_midi.Note(velocity=90, pitch=key + 21, start=s * dt, end=e * dt))
    pm.instruments.append(inst)
    return pm


def synthesize(pm: pretty_midi.PrettyMIDI, soundfont: Path, wav: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".mid") as f:
        pm.write(f.name)
        subprocess.run(
            ["fluidsynth", "-ni", "-g", "1.5", "-r", str(SAMPLE_RATE), "-F", str(wav),
             str(soundfont), f.name],
            check=True, capture_output=True,
        )  # fmt: skip


# Pressed keys light up green; keys that *should* be pressed but are not show blue.
_PRESSED_RGBA = (0.2, 0.8, 0.2, 1.0)
_TARGET_RGBA = (0.3, 0.55, 1.0, 1.0)
_DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)  # Lets the key's own white/black material show.


def _color_keys(physics, piano, pressed: np.ndarray, target: np.ndarray) -> None:
    rgba = np.tile(_DEFAULT_RGBA, (piano.n_keys, 1))
    rgba[target.astype(bool)] = _TARGET_RGBA
    rgba[pressed.astype(bool)] = _PRESSED_RGBA
    physics.bind(piano._key_geoms).rgba = rgba


def _caption(frame: np.ndarray, text: str) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    size = max(18, frame.shape[0] // 24)
    font = None
    for name in ("Helvetica.ttc", "DejaVuSans.ttf"):  # macOS, then common Linux font
        try:
            font = ImageFont.truetype(name, size)
            break
        except OSError:
            pass
    font = font or ImageFont.load_default(size=size)  # Pillow's built-in, any OS
    draw.rectangle([0, 0, frame.shape[1], int(size * 1.8)], fill=(0, 0, 0))
    draw.text((size // 2, int(size * 0.35)), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _env_config_for(args: RenderArgs) -> EnvConfig:
    """Rebuild the scene the recording came from: the joint positions only make sense in
    the same scene (e.g. a G1 run trained with the piano at 0.70 m)."""
    config_file = args.recording.parent.parent / "config.json"
    if not config_file.exists():
        return EnvConfig(midi_path=args.midi_path, embodiment=args.embodiment)
    saved = json.loads(config_file.read_text())["env"]
    known = {f.name for f in fields(EnvConfig)}
    return EnvConfig(**{k: v for k, v in saved.items() if k in known})


def render(args: RenderArgs) -> Path:
    rec = dict(np.load(args.recording))
    dt = float(rec["control_timestep"])
    out = args.out or args.recording.with_suffix(".mp4")
    soundfont = args.soundfont
    if soundfont is None:
        import robopianist

        soundfont = Path(robopianist.__file__).parent / "soundfonts" / "TimGM6mb.sf2"

    env = make_env(_env_config_for(args))
    env.reset()
    physics, piano = env.physics, env.task.piano
    step = int(rec.get("train_step", -1))
    # Caption with note F1 (each note struck once, on time) when the recording has it;
    # per-step F1 flatters jittery playing (see docs/06_results.md).
    if "metric_note_f1" in rec:
        score = f"note F1 {float(rec['metric_note_f1']):.2f}"
    else:
        score = f"F1 {float(rec.get('metric_f1', np.nan)):.2f}"
    text = f"training step {step:,}   {score}" if step >= 0 else score

    substeps = args.substeps
    frames = []
    qpos = rec["qpos"]
    for t in range(len(qpos)):
        nxt = qpos[min(t + 1, len(qpos) - 1)]
        for s in range(substeps):
            physics.data.qpos[:] = qpos[t] + (nxt - qpos[t]) * (s / substeps)
            physics.forward()
            _color_keys(physics, piano, rec["key_activation"][t], rec["goal"][t])
            frame = physics.render(args.height, args.width, camera_id=args.camera)
            frames.append(_caption(frame, text) if args.caption else frame)

    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(tmp) / "video.mp4"
        imageio.mimwrite(silent, frames, fps=substeps / dt, macro_block_size=1, quality=None,
                         output_params=["-crf", str(args.crf), "-preset", "medium", "-pix_fmt", "yuv420p"])  # fmt: skip
        wav = Path(tmp) / "audio.wav"
        synthesize(activations_to_midi(rec["key_activation"], dt), soundfont, wav)
        inputs = ["-i", str(silent), "-i", str(wav)]
        # Robot piano: pad with silence to the video's length, normalise loudness.
        robot = "[1:a]apad,loudnorm=I=-16:TP=-1.5[robot]"
        if args.backing_track is not None:
            from runaway.song import FIRST_STRIKE_IN_AUDIO_S

            # Start the record so its first strike lines up with the robot's first note.
            start = args.lead_in - FIRST_STRIKE_IN_AUDIO_S  # >0: delay the record
            delay_ms = max(0, round(1000 * start))
            skip = max(0.0, -start)
            inputs += ["-ss", f"{skip:.3f}", "-i", str(args.backing_track)]
            graph = (f"{robot};[2:a]adelay={delay_ms}|{delay_ms},volume={args.backing_volume}[song];"
                     "[robot][song]amix=inputs=2:duration=first:normalize=0[a]")  # fmt: skip
        else:
            graph = robot.replace("[robot]", "[a]")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", graph,
             "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", f"{args.audio_kbps}k",
             "-shortest", str(out)],
            check=True,
        )  # fmt: skip
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    render(tyro.cli(RenderArgs))
