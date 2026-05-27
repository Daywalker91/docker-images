# docker-images

Private Docker Image Repository für alle Custom-Images.  
Automatische Builds via GitHub Actions → ghcr.io.

## Images

| Image | Basis | Architekturen | Update-Trigger | Status |
|-------|-------|---------------|----------------|--------|
| [php-apache](./php-apache/) | `php:apache` | amd64, arm64, arm/v7 | täglich | ![Build php-apache](https://github.com/Daywalker91/docker-images/actions/workflows/php-apache.yml/badge.svg) |
| [media](./media/) | `python:slim` | amd64, arm64, arm/v7 | wöchentlich + yt-dlp Release | ![Build media](https://github.com/Daywalker91/docker-images/actions/workflows/media.yml/badge.svg) |
| [whisper](./whisper/) | `python:slim` + whisper.cpp | amd64, arm64 | wöchentlich + whisper.cpp Release | ![Build whisper](https://github.com/Daywalker91/docker-images/actions/workflows/whisper.yml/badge.svg) |
| [ollama](./ollama/) | `ollama/ollama` | amd64, arm64 | wöchentlich | ![Build ollama](https://github.com/Daywalker91/docker-images/actions/workflows/ollama.yml/badge.svg) |
| [rkllm](./rkllm/) | `python:slim` + rkllm Runtime | arm64 only | wöchentlich + rknn-llm Release | ![Build rkllm](https://github.com/Daywalker91/docker-images/actions/workflows/rkllm.yml/badge.svg) |

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
Enthält: `ffmpeg`, `yt-dlp`, Flask REST-API

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
Audio-Transkription via whisper.cpp (CPU).  
Enthält: whisper.cpp (kompiliert), Flask REST-API

**Endpunkte:**
- `POST /transcribe` – WAV → Transkript
- `GET /health`

**Env-Variablen:**
```
WHISPER_MODEL=tiny|base|small|medium   (Standard: base)
WHISPER_BACKEND=cpu|rknn               (Standard: cpu)
WHISPER_MODEL_DIR=/models
```

> **Hinweis:** RKNN-Support (NPU) ist geplant – Konvertierung via Google Colab (analog zu Obico).  
> Fertige RKNN-Modelle: [whisper-rknn-models Release](../../releases/tag/whisper-rknn-models)

```yaml
image: ghcr.io/daywalker91/whisper:latest
```

---

### ollama
Lokales LLM für Rezept-Extraktion (CPU/GPU).  
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

> **Hinweis:** Benötigt `/dev/rknpu` Device-Passthrough im k3s Manifest (via Generic Device Plugin).

```yaml
image: ghcr.io/daywalker91/rkllm:latest
```

---

## Automatische Updates

| Container | Schedule | Zusatz-Trigger |
|-----------|----------|----------------|
| php-apache | täglich 04:00 UTC | neuer `php:apache` Digest |
| media | montags 04:00 UTC | neue yt-dlp Version |
| whisper | montags 04:00 UTC | neue whisper.cpp Version |
| ollama | montags 04:00 UTC | neuer `ollama/ollama` Digest |
| rkllm | montags 04:00 UTC | neue rknn-llm Version |

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
    └─► rkllm-Container      (Rezept-Extraktion) :5002
    │         │
    │    Rezept → PHP → Review → DB
    │
    └─► ollama-Container     (Fallback CPU/GPU)  :11434
```

## GitHub Release Assets

| Release Tag | Inhalt |
|-------------|--------|
| `whisper-rknn-models` | Whisper Encoder + Decoder als `.rknn` für RK3588 |

## Setup (einmalig)

1. **GitHub Secret** `GH_PAT` anlegen:
   → Settings → Secrets → Actions → New repository secret
   → Fine-grained PAT mit Permissions: **Variables** (Read & Write) + **Contents** (Read & Write)

2. **Workflow permissions** auf "Read and write" stellen:
   → Settings → Actions → General → Workflow permissions

3. Beim ersten Workflow-Run werden alle Variablen automatisch erstellt.
