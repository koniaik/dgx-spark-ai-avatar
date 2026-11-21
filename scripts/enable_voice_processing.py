"""Enable browser voice processing (noise suppression + AGC) for the avatar mic.

OAC requests the mic with autoGainControl/noiseSuppression disabled (audio
fidelity choice). In a voice-agent deployment ambient noise (robot vacuum,
fans, TV) reaches SileroVAD, gets transcribed by whisper into single-word
hallucinations and the LLM dutifully answers them. Chrome's noise
suppression removes stationary hum before it ever hits the wire.

    python3 enable_voice_processing.py /path/to/OpenAvatarChat
"""

import sys
from pathlib import Path


def patch(repo: Path) -> None:
    p = repo / "src/handlers/client/rtc_client/client_handler_rtc.py"
    t = p.read_text()
    old = '"autoGainControl": False,\n                        "noiseSuppression": False,'
    new = (
        '"autoGainControl": True,\n'
        '                        "noiseSuppression": True,'
        '  # PATCH(pl-avatar): ambient noise (vacuum/fans) was transcribed as words'
    )
    if old in t:
        p.write_text(t.replace(old, new))
        print("patched: autoGainControl/noiseSuppression enabled")
    elif "PATCH(pl-avatar)" in t:
        print("already patched")
    else:
        print("PATTERN-MISS: constraints block not found")


if __name__ == "__main__":
    patch(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
