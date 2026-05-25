# docker-images

Private Docker Image Repository für alle Custom-Images.  
Automatische Builds via GitHub Actions → ghcr.io.

## Images

| Image | Basis | Architekturen | Status |
|-------|-------|---------------|--------|
| [php-apache](./php-apache/) | `php:apache` | amd64, arm64, arm/v7 | ![Build php-apache](https://github.com/Daywalker91/docker-images/actions/workflows/php-apache.yml/badge.svg) |

## Automatische Updates

Jedes Image wird **täglich um 04:00 UTC** auf ein neues Basis-Image geprüft.  
Wird eine neue Version erkannt → automatischer Rebuild & Push auf `ghcr.io`.

Ein Build wird außerdem ausgelöst bei:
- Änderung am jeweiligen `Dockerfile`
- Manuellem Start über "Run workflow"

## Setup (einmalig)

1. **GitHub Secret** `GH_PAT` anlegen:  
   → Settings → Secrets → Actions → New repository secret  
   → Fine-grained PAT mit Permission: **Actions Variables** (Read & Write)

2. Beim ersten Workflow-Run wird die Variable `PHP_APACHE_DIGEST` automatisch erstellt.

## Verwendung

```yaml
image: ghcr.io/daywalker91/php-apache:latest
```

## Geplante weitere Images

- `media` – ffmpeg + yt-dlp (Audio-Extraktion)
- `whisper` – whisper.cpp (Transkription)
- `llm` – Lokales LLM (Rezept-Extraktion)
