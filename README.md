# docker-images

Private Docker Image Repository für alle Custom-Images.  
Automatische Builds via GitHub Actions → ghcr.io.

## Images

| Image | Basis | Architekturen | Update-Trigger | Status |
|-------|-------|---------------|----------------|--------|
| [php-apache](./php-apache/) | `php:8.5-apache` | amd64, arm64, arm/v7 | täglich | ![Build php-apache](https://github.com/Daywalker91/docker-images/actions/workflows/php-apache.yml/badge.svg) |
| [media](./media/) | `python:3.13-slim` | amd64, arm64, arm/v7 | wöchentlich + yt-dlp Release | ![Build media](https://github.com/Daywalker91/docker-images/actions/workflows/media.yml/badge.svg) |
| [whisper](./whisper/) | `python:3.12-slim` + whisper.cpp | amd64, arm64 | wöchentlich + whisper.cpp Release | ![Build whisper](https://github.com/Daywalker91/docker-images/actions/workflows/whisper.yml/badge.svg) |
| [rk-llama](./rk-llama/) | `ubuntu:24.04` + rk-llama.cpp | arm64 only | wöchentlich + neuer rk-llama.cpp Commit | ![Build rk-llama](https://github.com/Daywalker91/docker-images/actions/workflows/rk-llama.yml/badge.svg) |
| [ollama](./ollama/) | `ollama/ollama` | amd64, arm64 | ⏸️ Workflow deaktiviert (nicht mehr im Einsatz) | ![Build ollama](https://github.com/Daywalker91/docker-images/actions/workflows/ollama.yml/badge.svg) |
| [rkllm](./rkllm/) | `python:slim` + rkllm Runtime | arm64 only | ⏸️ Workflow deaktiviert (nicht im Einsatz) | ![Build rkllm](https://github.com/Daywalker91/docker-images/actions/workflows/rkllm.yml/badge.svg) |

## Container-Übersicht

### php-apache
Custom PHP/Apache Image für die Rezept Bibliothek.  
Enthält: `mysqli`, `pdo`, `pdo_mysql`

```yaml
image: ghcr.io/daywalker91/php-apache:latest
```

---

### media
Audio-Extraktion aus lokalen Videos und Online-URLs.  
Enthält: `ffmpeg`, `yt-dlp` mit deno (YouTube) und curl-cffi (arm64/amd64), Flask REST-API

**Endpunkte:**
- `POST /extract/local` – lokale Datei → WAV 16kHz mono
- `POST /extract/url` – Online-URL → WAV 16kHz mono + Metadaten (Titel, Beschreibung)
- `GET /health`

**Env-Variablen:**
```
AUDIO_OUTPUT_DIR=/shared/audio
```

```yaml
image: ghcr.io/daywalker91/media:latest
```

---

### whisper
Audio-Transkription via whisper.cpp.  
Enthält: whisper.cpp Binary (CPU, auf arm64 mit ARMv8.2 dotprod/fp16), librknnrt (arm64), Flask REST-API

**Endpunkte:**
- `POST /transcribe` – `{"audio_path", "model", "backend", "language"}` → Job-ID
- `GET /job/<id>` – Status und Transkript
- `POST /preload` – Modell vorab herunterladen/laden
- `GET /health`

**Env-Variablen:**
```
WHISPER_MODEL=tiny|base|small|medium   (Standard: base)
WHISPER_BACKEND=cpu|rknn               (Standard: cpu)
WHISPER_THREADS=4
WHISPER_CPU_AFFINITY=4-7               (arm64: A76-Kerne; leer = keine Bindung)
WHISPER_BEAM_SIZE=5                    (1 = greedy, ~30 % schneller)
WHISPER_MODEL_DIR=/models
RKNN_MODEL_DIR=/models/rknn
```

**RKNN-Backend (NPU):**
- `backend: "rknn"` im Request (oder `WHISPER_BACKEND=rknn`), nur `base`; andere Größen laufen auf der CPU (die vorkonvertierten "tiny"-Modelle sind nur Englisch)
- Encoder + Decoder mit KV-Cache, Modelle von [sherpa-onnx](https://modelscope.cn/models/csukuangfj/2026-01-05-rknn) (rknn-toolkit2 2.1.0), daher librknnrt 2.1.0 – neuere Runtimes stürzen mit diesen Modellen ab
- Zugriff über die C-API mit Zero-Copy (`rknn_zc.py`) statt rknnlite
- Lange Aufnahmen in ≤ 30-s-Fenster an leisen Stellen geteilt, Sprache wird erkannt oder per `language` vorgegeben

**Messung RK1, 198 s deutsches Audio (2026-10-21):**

| Variante | Zeit | Wortfehler |
|---|---|---|
| CPU base | 31 s | 27,5 % |
| CPU base, `WHISPER_BEAM_SIZE=1` | 20 s | 30 % |
| CPU small | 82 s | 25,5 % |
| NPU base | ~95 s | 30 % |

Die NPU ist langsamer als die CPU, hält diese aber frei.

```yaml
image: ghcr.io/daywalker91/whisper:latest
```

---

### rk-llama
LLM-Inferenz für die Rezept-Bibliothek (Rezept-Extraktion, Übersetzung, Varianten) – der einzige LLM-Dienst im Cluster.
Enthält: [rk-llama.cpp](https://github.com/invisiofficial/rk-llama.cpp) (Branch `rknpu2`, llama.cpp mit RKNPU2-Backend), librknnrt, Flask-Wrapper mit OpenAI-kompatibler API

**arm64 only** – gebaut auf dem eigenen RK3588-Runner, läuft nur auf RK3588.

**Endpunkte:**
- `POST /v1/chat/completions`, `POST /v1/completions` – OpenAI-kompatibel; `model` = Dateiname auf dem Volume oder `hf:<owner>/<repo>/<datei>.gguf`
  - optional `rknpu_hybrid` – Rechenmodus pro Request (`CPU_STANDARD`, `W8A8_STANDARD`, `W8A8_HADAMARD`, kommagetrennt kombinierbar); ein anderer Modus oder ein anderes Modell startet llama-server neu (~30–55 s)
  - `response_format: {"type": "json_object"}` wird durchgereicht → reines JSON
- `GET /v1/models` – alle `*.gguf` auf dem Volume
- `GET /health` – geladenes Modell und Modus

**Env-Variablen:**
```
MODELS_DIR=/models
CONTEXT_SIZE=4096
LLAMA_THREADS=4
LLAMA_CPU_AFFINITY=4-7          (A76-Kerne; leer = keine Bindung)
RKNPU_HYBRID=W8A8_HADAMARD      (Standard, wenn der Request keinen Modus mitschickt)
```

Im Einsatz (Stand 2026-10-22): `qwen2.5-3b-instruct-q4_k_m.gguf` für die Extraktion, `gemma-3-4b-it-Q4_K_M.gguf` für Übersetzungen (auch Polnisch). Skaliert per KEDA auf 0 (Redis-Liste `rk-llama-queue`).

```yaml
image: ghcr.io/daywalker91/rk-llama:latest
```

---

### ollama
> ⏸️ Nicht mehr im Einsatz (2026-10-21 aus Cluster und App entfernt, rk-llama übernimmt). Image und Workflow bleiben, der Workflow ist deaktiviert.

Lokales LLM für Rezept-Extraktion (CPU/GPU, Fallback).  
Enthält: Ollama + konfigurierbares Modell

PHP spricht direkt mit der Ollama REST-API auf Port `11434`:
- `POST /api/generate`

**Env-Variablen:**
```
OLLAMA_MODEL=qwen2.5:3b   (oder llama3.2:3b, qwen2.5:7b, ...)
```

```yaml
image: ghcr.io/daywalker91/ollama:latest
```

---

### rkllm
> ⏸️ Nicht im Einsatz (2026-10-22 aus dem Cluster-Repository entfernt, rk-llama übernimmt). Image und Workflow bleiben, der Workflow ist deaktiviert.

LLM Inferenz auf dem RK3588 NPU via Rockchip RKLLM Runtime.  
Enthält: `librkllmrt.so`, Python ctypes-Wrapper, Flask REST-API

**arm64 only** – läuft nur auf RK3588 (RK1, Rock 5B, Orange Pi 5 etc.)

**Endpunkte:**
- `POST /generate` – Prompt → Antwort
- `GET /health`

**Env-Variablen:**
```
RKLLM_MODEL=1.5B|3B|7B              (Standard: 1.5B)
RKLLM_MODEL_DIR=/models
RKLLM_MAX_CONTEXT=4096
RKLLM_MAX_NEW_TOKENS=2048
RKLLM_TEMPERATURE=0.7
```

**Device-Passthrough (k3s):**  
Benötigt `/dev/rknpu` via Generic Device Plugin:
```yaml
resources:
  limits:
    rockchip.com/npu: 1
```

> **Hinweis:** Qwen2.5 RKLLM Konvertierungs-Notebook (Colab) ist noch ausstehend.  
> Aktuell werden fertige Modelle von HuggingFace (`c01zaut`) beim ersten Start heruntergeladen.

```yaml
image: ghcr.io/daywalker91/rkllm:latest
```

---

## Automatische Updates

| Container | Schedule | Zusatz-Trigger |
|-----------|----------|----------------|
| php-apache | täglich 04:00 UTC | neuer `php:8.5-apache` Digest |
| media | montags 04:00 UTC | neue yt-dlp Version |
| whisper | montags 04:00 UTC | neue whisper.cpp Version |
| rk-llama | montags 03:00 UTC | neuer Commit in rk-llama.cpp (Branch `rknpu2`) |
| ollama | – (Workflow deaktiviert) | – |
| rkllm | – (Workflow deaktiviert) | – |
| keepalive | monatlich am 1., 03:00 UTC | aktiviert alle Workflows neu, damit GitHub sie nicht nach 60 Tagen ohne Commit abschaltet (von Hand deaktivierte bleiben aus) |

## Shared Volume Flow

```
PHP-Container
    │
    ├─► media-Container      (ffmpeg + yt-dlp)   :5000
    │         │
    │    /shared/audio  ←──── Shared Volume
    │         │
    ├─► whisper-Container    (Transkription)      :5001
    │         │
    │    Transkript → PHP
    │
    └─► rk-llama-Container   (Extraktion, Übersetzung) :5002  ← CPU/NPU, KEDA
              │
         Rezept → PHP → Review → DB
```

## GitHub Release Assets

| Release Tag | Inhalt | Status |
|-------------|--------|--------|
| `qwen2.5-rkllm-models` | Qwen2.5-1.5B `.rkllm` für RK3588 | ⏸️ entfällt (rkllm nicht im Einsatz) |

Das frühere Release `whisper-rknn-models` (alter Whisper-Export ohne KV-Cache) wurde am 2026-10-22 gelöscht. Die NPU-Modelle für whisper kommen von [sherpa-onnx auf modelscope](https://modelscope.cn/models/csukuangfj/2026-01-05-rknn), siehe `whisper/download_rknn_models.sh`.

## Colab-Notebooks

| Notebook | Zweck | Status |
|----------|-------|--------|
| [whisper_rknn_convert.ipynb](./colab/whisper_rknn_convert.ipynb) | Whisper → ONNX → RKNN (alter Export ohne KV-Cache; das Release dazu ist gelöscht) | 📄 nur noch Dokumentation |
| qwen2.5_rkllm_convert.ipynb | Qwen2.5 → RKLLM Konvertierung | ⏳ ausstehend |

## Setup (einmalig)

1. **GitHub Secret** `GH_PAT` anlegen:
   → Settings → Secrets → Actions → New repository secret
   → Fine-grained PAT mit Permissions: **Variables** (Read & Write) + **Contents** (Read & Write)

2. **Workflow permissions** auf "Read and write" stellen:
   → Settings → Actions → General → Workflow permissions

3. Beim ersten Workflow-Run werden alle Variablen automatisch erstellt.
