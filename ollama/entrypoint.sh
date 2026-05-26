#!/bin/bash
set -e

MODEL=${OLLAMA_MODEL:-qwen2.5:3b}

echo "🚀 Starte Ollama Server..."
ollama serve &
SERVER_PID=$!

# Warten bis Ollama bereit ist
echo "⏳ Warte auf Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 1
done
echo "✅ Ollama bereit"

# Modell pullen falls nicht vorhanden
if ollama list | grep -q "^${MODEL}"; then
    echo "✅ Modell bereits vorhanden: ${MODEL}"
else
    echo "⬇️  Lade Modell: ${MODEL}"
    ollama pull "${MODEL}"
    echo "✅ Modell geladen: ${MODEL}"
fi

echo "🟢 Bereit – Modell: ${MODEL} | Port: 11434"

# Server im Vordergrund halten
wait $SERVER_PID
