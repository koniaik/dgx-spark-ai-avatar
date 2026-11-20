# Polish AI Avatar on NVIDIA DGX Spark

A fully local, real-time talking avatar that **speaks Polish** on a single NVIDIA DGX Spark (GB10: aarch64, CUDA 13, Blackwell sm_121, 128 GB unified memory).

The stack wires together three open-source projects:

| Component | Role | License |
|---|---|---|
| [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat) | Real-time conversation orchestration: VAD, barge-in/duplex, WebRTC UI, avatar rendering | Apache-2.0 |
| [Speaches](https://github.com/speaches-ai/speaches) | One container serving **ASR** (faster-whisper large-v3-turbo, Polish) and **TTS** (Piper, `pl_PL` voices) behind an OpenAI-compatible API | MIT |
| Your local LLM | Any OpenAI-compatible endpoint (vLLM, ollama, llama.cpp server, …) | — |

Everything runs on-device. No cloud APIs are involved in the speech path.

> Status: this exact sequence of steps was executed end-to-end on a DGX Spark (GB10) on 2026-11-20 — ASR, TTS, the LiteAvatar rendering worker and the web UI all run; the speech path is fully Polish. OpenAvatarChat's built-in ASR/TTS handlers do not support Polish (SenseVoice covers zh/en/ja/ko/yue only), so this repo adds two small custom handlers (`handlers/`) that call a Speaches server instead.

## Architecture

```
Microphone (browser, WebRTC, HTTPS :8899)
  → SileroVAD
  → SpeachesASR*  → faster-whisper large-v3-turbo (pl)  [Speaches container, :8000]
  → LLM (OpenAI-compatible API, e.g. http://127.0.0.1:8888/v1)
  → SpeachesTTS*  → Piper pl_PL voice                   [Speaches container, :8000]
  → Avatar handler → WebRTC video+audio back to the browser
                └─ * = custom handlers from this repo
```

Avatar variants (three example configs, one pipeline):

| Config | Avatar source | Notes |
|---|---|---|
| `configs/chat_pl_liteavatar.yaml` | Ready-made sample (LiteAvatar gallery) | **Start here** — works immediately, ~3 GB VRAM, 25 fps, Polish voice |
| `configs/chat_en_liteavatar.yaml` | Same, but the conversation runs in **English** | en_US Piper voice, English system prompt |
| `configs/chat_pl_musetalk.yaml` | Your own short video (1–5 min) | MuseTalk 1.5, per-avatar preprocessing on first start |
| `configs/chat_pl_flashhead.yaml` | Your own photo (zero-shot) | SoulX-FlashHead **Lite** (1.3 B DiT) — advanced: requires building `flash-attn` on aarch64, do this last |

Resource budget: the whole stack fits in roughly 6–10 GB of the 128 GB unified memory, leaving plenty of room for a co-located LLM server.

## Requirements

- NVIDIA DGX Spark (or any aarch64 Linux box with CUDA ≥ 13)
- Docker (Speaches publishes **linux/arm64** images; x86-only images will not run)
- [uv](https://docs.astral.sh/uv/) (installs the pinned Python 3.11 for you)
- An OpenAI-compatible LLM endpoint reachable from the machine
- A browser on any device in your LAN (UI is served over HTTPS with a self-signed cert)

## Quick start (validated on GB10)

Run everything as a regular user; no sudo needed.

```bash
# 0. tools
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# 1. OpenAvatarChat + the submodules this stack needs
git clone --depth 1 https://github.com/HumanAIGC-Engineering/OpenAvatarChat avatar
cd avatar
git submodule update --init --depth 1 \
  src/handlers/avatar/liteavatar/algo/liteavatar \
  src/handlers/vad/silerovad/silero_vad \
  src/service/frontend_service/frontend

# 2. drop in the Polish handlers, configs and scripts from this repo
cp -r /path/to/this-repo/handlers ./custom_handlers
cp -r /path/to/this-repo/configs ./configs-local
cp /path/to/this-repo/scripts/start.sh /path/to/this-repo/scripts/stop.sh .
chmod +x start.sh stop.sh

# 3. aarch64 patches (onnxruntime, nightly cu130 torch, triton override — see below)
python3 /path/to/this-repo/scripts/patch_aarch64.py .

# 4. dependencies + handler models (downloads torch nightly cu130 aarch64 etc.)
uv run install.py --config configs-local/chat_pl_liteavatar.yaml

# 5. undeclared runtime deps of the LiteAvatar algorithm submodule
uv pip install -r /path/to/this-repo/scripts/liteavatar_runtime_deps.txt

# 6. torchaudio (no stable aarch64 CUDA wheel; nightly cu130 ships aarch64)
uv pip install --no-deps --prerelease=allow \
  --index-url https://download.pytorch.org/whl/nightly/cu130 torchaudio

# 7. LiteAvatar sample weights (Paraformer + ONNX, ~1.3 GB, from ModelScope)
cd src/handlers/avatar/liteavatar/algo/liteavatar
PATH="$OLDPWD/.venv/bin:$PATH" bash download_model.sh
cd - >/dev/null

# 7b. self-signed certificate — upstream ships NO certs; without these the
#     service silently falls back to plain HTTP, and browsers refuse
#     microphone access (getUserMedia) on non-secure origins
openssl req -x509 -newkey rsa:2048 -sha256 -days 825 -nodes \
  -keyout ssl_certs/localhost.key -out ssl_certs/localhost.crt \
  -subj "/CN=<spark-ip>" \
  -addext "subjectAltName=IP:<spark-ip>,DNS:localhost"

# 8. Speaches — one container = ASR + TTS (arm64 image)
mkdir -p ~/speaches-cache
docker run -d --name speaches --restart unless-stopped \
  -p 8000:8000 \
  -v ~/speaches-cache:/home/ubuntu/.cache/huggingface/hub \
  ghcr.io/speaches-ai/speaches:latest-cpu

# 9. download the speech models (this Speaches release does not lazy-download)
curl -X POST http://127.0.0.1:8000/v1/models/speaches-ai/piper-pl_PL-darkman-medium
curl -X POST http://127.0.0.1:8000/v1/models/deepdml/faster-whisper-large-v3-turbo-ct2

# 10. point the configs at your LLM (api_url / model_name / api_key in
#     configs-local/chat_pl_*.yaml, section LLMOpenAICompatible), then run
./start.sh
```

Open `https://<spark-ip>:8899` in a browser, accept the self-signed certificate, allow the microphone and talk to the avatar in Polish. Stop with `./stop.sh`.

The handlers are registered through OpenAvatarChat's `handler_search_path` mechanism (`custom_handlers/` is listed in the example configs) — **no tracked file inside the OAC repository is modified**, so `git pull` upstream stays clean. The aarch64 patch edits two tracked files (`pyproject.toml`, `install.py`); re-running `scripts/patch_aarch64.py` is idempotent, and `git checkout -- pyproject.toml install.py` reverts it.

## What the aarch64 patch does (all verified against PyPI / download.pytorch.org metadata, 2026-11-20)

| Problem | Fix applied by `scripts/patch_aarch64.py` |
|---|---|
| `onnxruntime-gpu` has no Linux/aarch64 wheels (pinned in root + handlers) | swap to CPU `onnxruntime~=1.20.1`; `install.py`'s GPU replacement map flips direction on aarch64 (handlers fall back to the CPU execution provider) |
| PyPI / stable PyTorch indexes have **no CUDA wheels for aarch64** (PyPI's aarch64 torch is CPU-only; its `nvidia-*` deps are x86_64-gated) | point torch/torchvision at the **nightly cu130** index, which publishes `manylinux_2_28_aarch64` CUDA wheels (torch 2.16.0.dev+cu130 validated) |
| `torchaudio` has no aarch64 cu130 stable build | drop the root pin; installed separately from the nightly index with `--no-deps` |
| nightly torch needs `setuptools>=77` / `typing-extensions>=4.13`, upstream pins lower | widen both pins (setuptools kept `<81`: ≥81 removes `pkg_resources`, which the LiteAvatar algorithm imports) |
| nightly torch pins `triton==3.9.0+git<hash>`, which exists nowhere for cp311 | `override-dependencies = ["triton==3.9.0"]` (PyPI has an aarch64 wheel; irrelevant without `torch.compile`) |
| uv's universal lock also resolves for darwin, where no onnxruntime satisfies the pin | restrict `tool.uv.environments` to `linux && aarch64` |

## Service scripts

`scripts/start.sh` starts the Speaches container, waits for it to answer, then launches the demo in its own process group with a pidfile (never `pkill -f` over SSH — it self-matches). `scripts/stop.sh` reverses it, with a port-based fallback. Both were exercised through a dashboard launcher (start/stop buttons) during validation.

## Configuration reference (our handlers)

### `SpeachesASR`

| Field | Default | Meaning |
|---|---|---|
| `base_url` | `http://127.0.0.1:8000/v1` | Speaches API base |
| `model_name` | `deepdml/faster-whisper-large-v3-turbo-ct2` | ASR model id (must exist in the Speaches whisper registry) |
| `language` | `pl` | passed to the transcription request |
| `sample_rate` | `16000` | microphone audio rate |

### `SpeachesTTS`

| Field | Default | Meaning |
|---|---|---|
| `base_url` | `http://127.0.0.1:8000/v1` | Speaches API base |
| `model_name` | `speaches-ai/piper-pl_PL-darkman-medium` | Piper voice repo (registry shape: `speaches-ai/piper-<lang>_<REGION>-<voice>-<quality>`) |
| `voice` | `default` | required (non-empty) by the API; Piper derives the voice from the model |
| `sample_rate` | `24000` | output rate declared to the audio pipeline |

Other voices / better quality: F5-TTS Polish fine-tunes (CC-BY-NC base) or XTTS-v2 (non-commercial license) can be served behind the same OpenAI-compatible `/v1/audio/speech` contract — swapping them is a config change, not a code change.

## Switching the conversation language

The recipe ships with Polish as the default, but every part of the speech path is language-configurable in the yaml — nothing else changes:

| Component | What to change | Coverage |
|---|---|---|
| ASR (`SpeachesASR`) | `language: "pl"` → any faster-whisper language code (`"en"`, `"de"`, `"es"`, …). Set it to `""` to let faster-whisper **auto-detect**. | large-v3-turbo covers ~99 languages |
| TTS (`SpeachesTTS`) | `model_name` → any voice from the Speaches Piper registry; `sample_rate` → that voice's native rate (Piper "medium" voices are typically 22050 Hz) | **38 language packs / 124 voices** verified in the `speaches-ai` Piper registry (2026-11-20) |
| LLM replies | `system_prompt` → write it in the language you want the avatar to speak | depends on your LLM |
| Avatar rendering | nothing — the renderer is language-agnostic | — |

Popular Piper voices (id pattern: `speaches-ai/piper-<lang>_<REGION>-<voice>-<quality>`; all verified in the registry):

| Language | Voice id |
|---|---|
| Polish | `speaches-ai/piper-pl_PL-darkman-medium`, `speaches-ai/piper-pl_PL-gosia-medium` |
| English (US) | `speaches-ai/piper-en_US-amy-medium`, `speaches-ai/piper-en_US-ryan-medium`, `speaches-ai/piper-en_US-lessac-high` |
| English (UK) | `speaches-ai/piper-en_GB-alan-medium` |
| German | `speaches-ai/piper-de_DE-thorsten-medium` |
| French | `speaches-ai/piper-fr_FR-gilles-low`, `speaches-ai/piper-fr_FR-mls-medium` |
| Spanish | `speaches-ai/piper-es_ES-davefx-medium`, `speaches-ai/piper-es_MX-claude-high` |
| Italian | `speaches-ai/piper-it_IT-paola-medium` |
| Russian | `speaches-ai/piper-ru_RU-denis-medium` |
| Ukrainian | `speaches-ai/piper-uk_UA-ukrainian_tts-medium` |
| Czech / Slovak | `speaches-ai/piper-cs_CZ-jirka-medium`, `speaches-ai/piper-sk_SK-lili-medium` |
| Chinese (Mandarin) | `speaches-ai/piper-zh_CN-huayan-medium` |

Full list: <https://huggingface.co/speaches-ai> (filter `piper-`), or query the running container. After picking a voice, download it (this Speaches release does not lazy-download):

```bash
curl -X POST http://127.0.0.1:8000/v1/models/speaches-ai/piper-en_US-amy-medium
```

The easiest way to test a second language is `configs/chat_en_liteavatar.yaml` — it differs from the Polish one only in `language`, `model_name`/`sample_rate` (TTS) and `system_prompt`.

## Choosing the LLM — local or cloud

The `LLMOpenAICompatible` handler speaks plain OpenAI-compatible `/v1/chat/completions`, so any endpoint of that shape works. Three fields decide everything (`configs-local/chat_pl_*.yaml`, section `LLMOpenAICompatible`):

```yaml
        model_name: "<model id>"
        api_url: "<base url ending in /v1>"
        api_key: "<key; any non-empty string for local servers>"
```

**Local servers** (no data leaves the machine):

| Server | `api_url` |
|---|---|
| vLLM | `http://localhost:8000/v1` |
| llama.cpp server | `http://localhost:8080/v1` |
| ollama | `http://localhost:11434/v1` |
| LM Studio | `http://localhost:1234/v1` |

**Cloud providers** (any OpenAI-compatible API — note the conversation *text* then leaves your machine; the audio path stays local):

| Provider | `api_url` | `model_name` example |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| OpenRouter | `https://openrouter.ai/api/v1` | provider-prefixed id, e.g. `meta-llama/llama-3.3-70b-instruct` |
| Groq | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| Mistral | `https://api.mistral.ai/v1` | `mistral-small-latest` |
| DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
| Google Gemini (OpenAI-compat) | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` |

Notes:
- `api_key` is **required** by the handler — for local servers without auth put any non-empty string (e.g. `"EMPTY"`). If the field is omitted entirely, the handler falls back to the `DASHSCOPE_API_KEY` environment variable (a legacy OpenAvatarChat default); setting the field explicitly is the recommended path.
- The example configs live under `configs-local/` (untracked) — never commit real API keys anywhere in the repo.
- Switching LLM or language requires only an app restart (`./stop.sh && ./start.sh` or the dashboard launcher buttons), not a reinstall.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `uv run install.py` fails on `onnxruntime-gpu` resolution | run `scripts/patch_aarch64.py` first (step 3) |
| `Failed to import module asr/speaches/...` | handler directories must not shadow OAC's own top-level packages (`asr`, `tts`); this repo uses `pl_speaches_asr` / `pl_speaches_tts` for exactly that reason |
| `Error in avatar process: No module named 'pkg_resources'` | setuptools ≥ 81 removes it — keep the patched `<81` pin |
| `Error in avatar process: No module named 'typeguard'` or `argument "default" (None) is not an instance of str` | LiteAvatar's algorithm targets typeguard 2.x — install `typeguard==2.13.3` (already pinned in `scripts/liteavatar_runtime_deps.txt`) |
| `Error in avatar process: No module named 'torchaudio'` | step 6 (`--no-deps` from the nightly cu130 index) |
| `No such file or directory: './weights/.../model.pb'` | step 7 — run `download_model.sh` from inside `algo/liteavatar/` with the main venv's `bin` on PATH (do **not** run it with `uv run` from the submodule: its own `pyproject.toml` pins x86-only `onnxruntime-gpu` and uv would build a broken venv there) |
| `Model '...' is not installed locally` from Speaches | download explicitly: `curl -X POST http://127.0.0.1:8000/v1/models/<model_id>` (this release does not lazy-download) |
| Whisper model id unknown | list available ids: the Speaches whisper registry exposes HF repos such as `Systran/faster-whisper-large-v3` or `deepdml/faster-whisper-large-v3-turbo-ct2` |
| Browser microphone blocked / `ERR_SSL_PROTOCOL_ERROR` on `https://` | check the startup log: if it says `Uvicorn running on http://` there is no cert — the upstream repo ships an **empty** `ssl_certs/` directory (only a placeholder file) and the service silently falls back to plain HTTP. Generate one (Quick start step 7b), restart, and expect `SSL enabled.` + `Uvicorn running on https://` in the log. Plain HTTP is only usable from `localhost`, which a LAN IP never is |
| Server log: `Cert file ... not found` warning at startup | same as above — the cert files were not generated |
| x86-only Docker images won't run | GB10 Docker runs arm64 images only (e.g. NVIDIA ACE NIMs and Duix.Avatar images are amd64-only — dead ends on this hardware) |
| Triton `ptxas` from CUDA 12.x doesn't know sm_121 | point `TRITON_PTXAS_PATH` at CUDA 13's `ptxas` |

## Credits

- [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat) (HumanAIGC-Engineering) — the conversation framework
- [Speaches](https://github.com/speaches-ai/speaches) — ASR/TTS server
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) / [deepdml large-v3-turbo-ct2](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2) — Polish-capable ASR
- [Piper](https://github.com/rhasspy/piper) / [speaches-ai piper voices](https://huggingface.co/speaches-ai) — Polish TTS
- [LiteAvatar](https://github.com/HumanAIGC/lite-avatar) (HumanAIGC) — the sample avatar renderer
- [MuseTalk](https://github.com/TMElyralab/MuseTalk) — video-driven avatar variant
- [SoulX-FlashHead](https://github.com/Soul-AILab/SoulX-FlashHead) (Soul-AILab) — photo-driven avatar variant

## License

MIT — see [LICENSE](LICENSE). The upstream projects keep their own licenses (see table above and their repos).
