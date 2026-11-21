"""Repair the filter_text patch on the Spark: clean, correctly-indented, single backslashes."""

import sys
from pathlib import Path

CLEAN = (
    "def filter_text(text):\n"
    "    # PATCH(pl-avatar): upstream kept only ASCII + CJK, deleting Polish diacritics\n"
    '    # from what the LLM sees ("czes??" -> "cze"). Keep all Unicode word characters.\n'
    '    pattern = r"[^\\w\\s,.~!?()\\-:;%\u2026\u2014\u201d\u201c\u2018\u2019\uff0c\u3002\uff01\uff1f]+"\n'
    "    filtered_text = re.sub(pattern, \"\", text)\n"
    "    return filtered_text\n"
)


def patch(repo: Path) -> None:
    p = repo / "src/handlers/llm/openai_compatible/chat_history_manager.py"
    t = p.read_text(encoding="utf-8")
    start = t.index("def filter_text(text):")
    end = t.index("\n\n\n", start) + 1  # up to the blank separation before class
    p.write_text(t[:start] + CLEAN + t[end:], encoding="utf-8")
    # sanity: compiles and keeps Polish
    import re as _re
    pat = r"[^\w\s,.~!?()\-:;%\u2026\u2014\u201d\u201c\u2018\u2019\uff0c\u3002\uff01\uff1f]+"
    ok = _re.sub(pat, "", "cześć, w pełni poprawę! 🎉") == "cześć, w pełni poprawę! "
    print("patched; regex sanity:", "OK" if ok else "FAIL")


if __name__ == "__main__":
    patch(Path(sys.argv[1]))
