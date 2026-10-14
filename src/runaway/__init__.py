"""Teaching simulated robot hands (and later a humanoid) to play Kanye West's "Runaway"."""

import os
import sys

# RoboPianist imports pyfluidsynth (its audio synthesizer) at package import time.
# On macOS, Homebrew puts libfluidsynth in /opt/homebrew/lib, which Python's library
# lookup does not search by default. On Linux, `apt install libfluidsynth3` is enough.
if sys.platform == "darwin" and os.path.isdir("/opt/homebrew/lib"):
    os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
