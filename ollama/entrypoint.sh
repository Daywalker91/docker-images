#!/bin/bash
set -e

echo "🚀 Starte Ollama Server..."
ollama serve &
SERVER_PID=$!

echo "⏳ Warte auf Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "✅ Ollama bereit – Modell wird beim ersten Request automatisch geladen"

wait $SERVER_PID
