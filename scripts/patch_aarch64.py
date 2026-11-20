#!/usr/bin/env python3
"""Patch OpenAvatarChat for aarch64 hosts (NVIDIA DGX Spark / GB10).

Four upstream assumptions break on aarch64 (each verified against PyPI /
download.pytorch.org metadata on 2026-11-20):

1. `onnxruntime-gpu` publishes no Linux/aarch64 wheels -> swap to CPU
   `onnxruntime` (pin ~=1.20.1; 1.20.2 has no aarch64 wheel either).
2. PyPI / stable PyTorch indexes ship no CUDA wheels for aarch64 (PyPI's
   aarch64 torch is CPU-only: its nvidia-* deps are x86_64-gated) -> point
   torch/torchvision at the nightly cu130 index, which publishes
   manylinux_2_28_aarch64 CUDA wheels. `torchaudio` has no aarch64 cu130
   build and nothing in OAC imports it -> drop the dependency.
3. uv's universal lock resolves for darwin too, where no onnxruntime version
   satisfies the pin -> restrict resolution environments to linux aarch64.
4. install.py would otherwise force x86_64 pins back in
   (PACKAGE_REPLACEMENTS maps onnxruntime->onnxruntime-gpu, PROTECTED_PACKAGES
   pins torch==2.8.0) -> make both platform-conditional.

Usage: python3 patch_aarch64.py /path/to/OpenAvatarChat
Idempotent: running it twice is a no-op. If pyproject.toml was corrupted by an
older revision of this script, restore it first with
`git checkout -- pyproject.toml`.
"""

import sys
from pathlib import Path

NIGHTLY_INDEX_BLOCK = (
    '[[tool.uv.index]]\n'
    'name = "pytorch-cu130-nightly"\n'
    'url = "https://download.pytorch.org/whl/nightly/cu130"\n'
    'explicit = true'
)
ENV_LINE = 'environments = ["sys_platform == \'linux\' and platform_machine == \'aarch64\'"]'


def patch_pyproject(repo: Path) -> int:
    pyproject = repo / "pyproject.toml"
    text = pyproject.read_text()

    if "import argparse" in text.splitlines()[:5].__str__():
        print("ERROR: pyproject.toml looks corrupted (contains install.py text);")
        print("       restore it first: git checkout -- pyproject.toml")
        return 0

    orig_text = text

    # 1. onnxruntime
    if '"onnxruntime-gpu~=1.20.2"' in text:
        text = text.replace('"onnxruntime-gpu~=1.20.2"', '"onnxruntime~=1.20.1"')
    elif '"onnxruntime~=1.20.2"' in text:  # older revision of this patch
        text = text.replace('"onnxruntime~=1.20.2"', '"onnxruntime~=1.20.1"')
    if '"onnxruntime~=1.20.1"' in text:
        pass
    else:
        print("WARNING: onnxruntime pin not found in pyproject.toml (upstream changed?)")

    # 2b. nightly torch requires setuptools>=77 / typing-extensions>=4.13; upstream pins lower
    text = text.replace('"setuptools>=69.5.1,<70"', '"setuptools>=77.0.3,<81"')  # >=81 drops pkg_resources (lite-avatar algo needs it)
    if '"setuptools>=77.0.3,<81"' not in text:
        text = text.replace('"setuptools>=69.5.1"', '"setuptools>=77.0.3,<81"')  # earlier revision of this patch
    text = text.replace('"typing-extensions>=4.12.2,<4.13.dev0"', '"typing-extensions>=4.13"')
    text = text.replace('"typing-extensions~=4.12.2"', '"typing-extensions>=4.13"')

    # 2. torch stack -> nightly cu130 index
    if '"pytorch-cu130-nightly"' not in text:
        text = text.replace('"torch==2.8.0",', '"torch>=2.9",')
        text = text.replace('"torchvision==0.23.0",', '"torchvision>=0.24",')
        text = text.replace('    "torchaudio==2.8.0",\n', "")
        for pkg in ("torch", "torchvision"):
            text = text.replace(
                pkg + ' = [\n    { index = "pytorch-cu128" },\n]',
                pkg + ' = [\n    { index = "pytorch-cu130-nightly" },\n]',
            )
        text = text.replace(
            'torchaudio = [\n    { index = "pytorch-cu128" },\n]\n',
            "",
        )
        text = text.replace(
            '[[tool.uv.index]]\nname = "pytorch-cu128"\nurl = "https://download.pytorch.org/whl/cu128"\nexplicit = true',
            NIGHTLY_INDEX_BLOCK,
        )

    # 3. restrict uv resolution environments
    # 4. nightly torch pins triton==3.9.0+git<hash>, which exists nowhere for cp311
    #    (nightly index ships cp314/cp315 only) -> override to PyPI triton==3.9.0
    #    (aarch64 wheel exists; torch without torch.compile tolerates the exact-version drift)
    if ENV_LINE in text and '"triton==' not in text:
        text = text.replace(
            ENV_LINE,
            ENV_LINE + '\noverride-dependencies = ["triton==3.9.0"]',
        )
    elif "environments = " in text:
        print("WARNING: pyproject already restricts tool.uv environments (unexpected value)")
    else:
        block = (
            "\n[tool.uv]\n"
            "# Patched for aarch64 (DGX Spark): other resolution splits have no\n"
            "# onnxruntime/torch wheels that satisfy the requirements.\n"
            + ENV_LINE + "\n"
            'override-dependencies = ["triton==3.9.0"]\n'
        )
        text = text.rstrip("\n") + "\n" + block

    if text != orig_text:
        pyproject.write_text(text)
        print("patched pyproject.toml (onnxruntime / nightly cu130 torch / environments)")
        return 1
    print("pyproject.toml already patched")
    return 0


def patch_install(repo: Path) -> int:
    install = repo / "install.py"
    text = install.read_text()
    changed = False

    old_repl = 'PACKAGE_REPLACEMENTS = {\n    "onnxruntime": "onnxruntime-gpu",\n}'
    new_repl = (
        "import platform\n\n"
        "# aarch64 (DGX Spark/GB10): onnxruntime-gpu has no wheels for this platform,\n"
        "# so the replacement direction flips and handlers use the CPU execution provider.\n"
        "PACKAGE_REPLACEMENTS = (\n"
        '    {"onnxruntime": "onnxruntime-gpu"}\n'
        '    if platform.machine() != "aarch64"\n'
        '    else {"onnxruntime-gpu": "onnxruntime"}\n'
        ")"
    )
    if old_repl in text:
        text = text.replace(old_repl, new_repl)
        changed = True

    old_protected = (
        "PROTECTED_PACKAGES = {\n"
        '    "torch": "==2.8.0",\n'
        '    "torchvision": "==0.23.0",\n'
        '    "torchaudio": "==2.8.0",\n'
        "}"
    )
    new_protected = (
        "import platform as _platform\n\n"
        "# aarch64: torch comes from the nightly cu130 index (no stable aarch64 CUDA wheels),\n"
        "# so the handler-dependency phase must not force the x86_64 pins back in.\n"
        "PROTECTED_PACKAGES = (\n"
        "    {\n"
        '        "torch": "==2.8.0",\n'
        '        "torchvision": "==0.23.0",\n'
        '        "torchaudio": "==2.8.0",\n'
        "    }\n"
        '    if _platform.machine() != "aarch64"\n'
        "    else {\n"
        '        "torch": ">=2.9",\n'
        '        "torchvision": ">=0.24",\n'
        "    }\n"
        ")"
    )
    if old_protected in text:
        text = text.replace(old_protected, new_protected)
        changed = True

    if changed:
        install.write_text(text)
        print("patched install.py (PACKAGE_REPLACEMENTS / PROTECTED_PACKAGES for aarch64)")
        return 1
    if 'platform.machine() != "aarch64"' in text:
        print("install.py already patched")
        return 0
    print("WARNING: install.py blocks not found (upstream changed?)")
    return 0


if __name__ == "__main__":
    repo_path = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    n = patch_pyproject(repo_path) + patch_install(repo_path)
    print(f"done ({n} file(s) changed)")
    sys.exit(0)
