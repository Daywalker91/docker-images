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

## Container-Übersicht

### php-apache
Custom PHP/Apache Image für die Rezept Bibliothek.  
Enthält: `mysqli`, `pdo`, `pdo_mysql`

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

### whisper
Audio-Transkription via whisper.cpp.  
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

> **Hinweis:** RKNN-Support (NPU) ist noch nicht implementiert – geplant via Google Colab.

### ollama
Lokales LLM für Rezept-Extraktion aus Transkripten.  
Enthält: Ollama + konfigurierbares Modell

PHP spricht direkt mit der Ollama REST-API:
- `POST http://ollama:11434/api/generate`

**Env-Variablen:**
```
OLLAMA_MODEL=qwen2.5:3b   (oder llama3.2:3b, qwen2.5:7b, ...)
```

## Automatische Updates

Jeder Container prüft wöchentlich (montags 04:00 UTC) auf neue Basis-Image-Versionen.  
`php-apache` prüft täglich. `media` und `whisper` prüfen zusätzlich auf neue Releases von yt-dlp bzw. whisper.cpp.

Ein Build wird außerdem ausgelöst bei:
- Änderung am jeweiligen `Dockerfile` oder den zugehörigen Dateien
- Manuellem Start über "Run workflow"

## Setup (einmalig)

1. **GitHub Secret** `GH_PAT` anlegen:  
   → Settings → Secrets → Actions → New repository secret  
   → Fine-grained PAT mit Permission: **Variables** (Read & Write)

2. **Workflow permissions** auf "Read and write" stellen:  
   → Settings → Actions → General → Workflow permissions

3. Beim ersten Workflow-Run werden alle Variablen automatisch erstellt.

## Verwendung

```yaml
image: ghcr.io/daywalker91/php-apache:latest
image: ghcr.io/daywalker91/media:latest
image: ghcr.io/daywalker91/whisper:latest
image: ghcr.io/daywalker91/ollama:latest
```

## Shared Volume Flow

```
PHP-Container
    │
    ├─► media-Container    (ffmpeg + yt-dlp)
    │         │
    │    /shared/audio  (Shared Volume)
    │         │
    ├─► whisper-Container  (Transkription)
    │         │
    │    Transkript → PHP
    │
    └─► ollama-Container   (Rezept-Extraktion)
              │
         Rezept → PHP → Review → DB
```
