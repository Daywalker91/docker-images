# Base: offizielles Ollama Image
FROM ollama/ollama

LABEL org.opencontainers.image.source="https://github.com/Daywalker91/docker-images"
LABEL org.opencontainers.image.description="Ollama LLM container – Rezept-Extraktion"
LABEL org.opencontainers.image.licenses="MIT"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Modell-Verzeichnis (als Volume mounten)
VOLUME /root/.ollama

EXPOSE 11434

ENTRYPOINT ["/entrypoint.sh"]
