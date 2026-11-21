"""Fix OAC's LLM input filter destroying non-ASCII/non-Chinese diacritics.

chat_history_manager.filter_text keeps only [a-zA-Z0-9], basic punctuation
and CJK — every Polish/German/French... diacritic is deleted from the user's
text and chat history BEFORE the request reaches the LLM ("cześć" -> "cze").
The model then echoes diacritic-less text and the TTS speaks mangled words.

Replacement keeps all Unicode word characters (letters+digits in any script),
whitespace and common punctuation; emoji/markdown symbols are still stripped.

    python3 fix_polish_filter.py /path/to/OpenAvatarChat
"""

import sys
from pathlib import Path

OLD = 'pattern = r"[^a-zA-Z0-9\\u4e00-\\u9fff,.\\~!?，。！？ ]"  # 匹配不在范围内的字符'
NEW = (
    "# PATCH(pl-avatar): upstream kept only ASCII + CJK, deleting Polish (and every other)\n"
    "# script's diacritics from what the LLM sees (\"cześć\" -> \"cze\", model then replies\n"
    "# without diacritics). Keep all Unicode word characters instead; emoji still stripped.\n"
    'pattern = r"[^\\w\\s,.\\~!?，。！？…()\\-:;\\u2014\\u201d\\u201c\\u2018\\u2019%]+"'
)


def patch(repo: Path) -> None:
    p = repo / "src/handlers/llm/openai_compatible/chat_history_manager.py"
    t = p.read_text(encoding="utf-8")
    if "PATCH(pl-avatar)" in t:
        print("already patched")
        return
    if OLD not in t:
        print("PATTERN-MISS: filter_text pattern not found (upstream changed?)")
        return
    p.write_text(t.replace(OLD, NEW, 1), encoding="utf-8")
    print("patched: filter_text now Unicode-aware (Polish diacritics preserved)")


if __name__ == "__main__":
    patch(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
