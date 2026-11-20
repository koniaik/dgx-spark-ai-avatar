"""Disable OAC's H264-first codec ordering (client_handler_rtc.py).

H264-first SDP answers occasionally fail browser parsing
("Failed to parse SessionDescription", esp. on reconnects). Keeping aiortc's
default order (VP8 first) is universally accepted. Run on the OAC checkout:

    python3 disable_h264_priority.py /path/to/OpenAvatarChat
"""

import sys
from pathlib import Path


def patch(repo: Path) -> None:
    p = repo / "src/handlers/client/rtc_client/client_handler_rtc.py"
    t = p.read_text()
    if "PATCH(pl-avatar)" in t:
        print("already patched")
        return
    old = "_prioritize_h264()\n"
    new = (
        "# PATCH(pl-avatar): H264-first answers occasionally fail browser SDP parsing\n"
        '# ("Failed to parse SessionDescription", esp. on reconnects) - keep aiortc\'s\n'
        "# default codec order (VP8 first), which every browser accepts.\n"
        "# _prioritize_h264()\n"
    )
    if old in t:
        p.write_text(t.replace(old, new, 1))
        print("patched: H264 priority disabled (VP8-first)")
    else:
        print("PATTERN-MISS: _prioritize_h264() call not found")


if __name__ == "__main__":
    patch(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
