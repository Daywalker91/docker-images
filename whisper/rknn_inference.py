"""
Whisper RKNN Inferenz via rknnlite.
Nutzt die konvertierten Encoder + Decoder .rknn Modelle vom GitHub Release.
"""
import logging
import numpy as np

log = logging.getLogger(__name__)

# Modell-Dimensionen pro Größe
MODEL_DIMS = {
    "tiny":   {"n_audio_state": 384,  "n_text_ctx": 448},
    "base":   {"n_audio_state": 512,  "n_text_ctx": 448},
    "small":  {"n_audio_state": 768,  "n_text_ctx": 448},
    "medium": {"n_audio_state": 1024, "n_text_ctx": 448},
}


class WhisperRKNN:
    """Whisper Inferenz auf RK3588 NPU via rknnlite."""

    def __init__(self, encoder_path: str, decoder_path: str, model_size: str = "base"):
        from rknnlite.api import RKNNLite

        dims = MODEL_DIMS.get(model_size, MODEL_DIMS["base"])
        self.n_audio_state = dims["n_audio_state"]
        self.n_text_ctx    = dims["n_text_ctx"]
        self.model_size    = model_size

        log.info(f"Lade RKNN Encoder: {encoder_path}")
        self.encoder = RKNNLite()
        ret = self.encoder.load_rknn(encoder_path)
        assert ret == 0, f"Encoder load_rknn fehlgeschlagen: {ret}"
        ret = self.encoder.init_runtime()
        assert ret == 0, f"Encoder init_runtime fehlgeschlagen: {ret}"

        log.info(f"Lade RKNN Decoder: {decoder_path}")
        self.decoder = RKNNLite()
        ret = self.decoder.load_rknn(decoder_path)
        assert ret == 0, f"Decoder load_rknn fehlgeschlagen: {ret}"
        ret = self.decoder.init_runtime()
        assert ret == 0, f"Decoder init_runtime fehlgeschlagen: {ret}"

        log.info(f"✅ RKNN Modell bereit (whisper-{model_size})")

    def transcribe(self, wav_path: str, language: str = "auto") -> str:
        """
        WAV-Datei → Transkript via RKNN NPU.
        language: ISO-639-1 Code (z.B. 'de', 'en') oder 'auto' für Erkennung.
        """
        import whisper

        # ── Audio → Mel-Spektrogramm ──────────────────────────────────────────
        audio = whisper.load_audio(wav_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).numpy()
        mel = mel[np.newaxis, :].astype(np.float32)  # (1, 80, 3000)

        # ── Encoder ───────────────────────────────────────────────────────────
        encoder_output = self.encoder.inference(inputs=[mel])[0]  # (1, 1500, n_audio_state)
        encoder_output = encoder_output.astype(np.float32)

        # ── Tokenizer ─────────────────────────────────────────────────────────
        tokenizer = whisper.tokenizer.get_tokenizer(
            multilingual=True,
            language=None if language == "auto" else language,
            task="transcribe",
        )
        initial_tokens = list(tokenizer.sot_sequence_including_notimestamps)
        tokens = initial_tokens.copy()

        # ── Decoder Loop (greedy) ─────────────────────────────────────────────
        max_new_tokens = self.n_text_ctx // 2  # max 224 neue Tokens

        for _ in range(max_new_tokens):
            # Tokens auf n_text_ctx padden (statische Shape des RKNN-Modells)
            padded = np.zeros((1, self.n_text_ctx), dtype=np.int64)
            padded[0, :len(tokens)] = tokens

            # Decoder ausführen
            logits = self.decoder.inference(inputs=[padded, encoder_output])[0]
            # logits: (1, n_text_ctx, vocab_size)

            # Logits an aktueller Position
            pos = len(tokens) - 1
            next_token = int(np.argmax(logits[0, pos, :]))

            if next_token == tokenizer.eot:
                break

            tokens.append(next_token)

        # ── Tokens → Text ──────────────────────────────────────────────────────
        text_tokens = [t for t in tokens[len(initial_tokens):] if t < tokenizer.eot]
        text = tokenizer.decode(text_tokens).strip()

        log.info(f"RKNN Transkription: {len(text)} Zeichen")
        return text

    def release(self):
        self.encoder.release()
        self.decoder.release()
        log.info("RKNN Modell freigegeben")
