# TODO

Status on 2026-10-12 (night): the two-hand setup plays the Runaway riff (note F1
0.99–1.00). The G1 humanoid plays it too since round 7 (best note F1 0.905: 43/44 strikes on
time, 51 presses for 44 strikes), after fixing a bug that hid the fingers from the policy.
See [docs/06_results.md](docs/06_results.md).

## 1. Make the G1 play cleanly (close the gap to the hands: 0.90 → 0.99)
Done: IK control, hovering hands, extra-press penalty, arm filter, finger observations
(the real fix), rounds 3–7 exported to the bucket. Remaining precision gap: 7–30 extra presses.
- [ ] Train longer (8M) from the best round-7 checkpoints (several runs still improving at 4M).
- [ ] Extra-press penalty from scratch with 3 seeds (one seed gave the best run, 0.905).
- [ ] Ramp the extra-press penalty in after the notes are learned (curriculum).
- [ ] Select checkpoints by note F1; evaluate several episodes per checkpoint (eval is noisy).
- [ ] Optional warm start: initialise the G1 finger policy from the round-1 hands policy.

## 2. Ablations (humanoid task; the hands task saturates at 1.0 for everything)
Each with ≥ 3 seeds per setting; one 8-run sweep on an 8×H100 node ≈ $25.
- [ ] **Arm action range**: ±0.25 / ±0.4 / ±0.8 / ±1.2 rad (we have one striking data
      point: ±1.2 learned nothing in 300k steps, ±0.4 learns).
- [ ] **Joint commands vs hand-position commands through IK** (policy outputs a palm target,
      IK, e.g. `mink`, turns it into shoulder/elbow targets).
- [ ] **Automatic (OT) fingering on vs off.**
- [ ] **Algorithms**: SAC (ours) vs PPO vs FastTD3 (and FastSAC), same budget, same seeds.
- [ ] Report note F1 *and* per-step F1 for all of them.

## 3. More songs
- [ ] Make the transcription pipeline song-agnostic (`scripts/transcribe_audio.py` →
      `song.py`-style data file per song; currently Runaway's riff is hand-structured).
- [ ] Try songs of increasing difficulty: chords, both hands, faster tempo, wider jumps.
- [ ] Also run public-domain pieces from RoboPianist's repertoire so results compare with
      published numbers (and can be released without copyright questions).
- [ ] A small "song library": one trained policy + video per song.

## 4. Engineering nice-to-haves
- [ ] GPU-parallel simulation (MuJoCo Warp / MJX) + FastSAC: from ~1.5k to ~100k steps/s.
- [ ] Ask SF Compute for "GPU graphics" so videos render on the node (EGL) instead of the Mac.
- [ ] Make `sync_dashboard.sh` incremental (only pull new evals).
- [ ] Unit test for `humanoid.py` mount geometry (hands point at the keys, palms down).
- [ ] Better piano soundfont for the videos (Salamander Grand, 1.3 GB, optional).
