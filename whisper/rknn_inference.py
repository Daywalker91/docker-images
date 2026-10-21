"""
Whisper RKNN Inferenz auf dem RK3588 – Encoder/Decoder mit KV-Cache.

Modelle: eigener Export auf Basis von sherpa-onnx (scripts/whisper/rknn), siehe
whisper/rknn-export/. Encoder: mel (1, 80, 3000) -> cross_k/cross_v je
Decoder-Schicht. Decoder: EIN Token pro Aufruf + Self-KV-Cache (448 Plätze) +
Cross-KV + offset + mask -> logits + neue Self-K/V-Zeile. Ein Decoder-Schritt
ist damit konstant teuer statt mit der Textlänge zu wachsen.

Die Modelle laufen über rknn_zc (C-API, Zero-Copy) statt rknnlite: rknnlite
kopiert und konvertiert pro Token alle Eingaben (~48 MB bei base), obwohl sich
nur eine Cache-Zeile ändert.

Lange Aufnahmen werden in Fenster von höchstens 30 s geteilt, bevorzugt an
leisen Stellen. Audio-Vorverarbeitung und Tokenizer kommen aus whisper_audio.py
(1:1 wie openai-whisper, ohne PyTorch).
"""
import logging
import re
import time

import numpy as np

import rknn_zc
import whisper_audio as whisper

log = logging.getLogger(__name__)

N_TEXT_CTX = 448
SR = whisper.SAMPLE_RATE
MAX_WINDOW_S = 30.0       # Encoder-Eingabe ist fest 30 s
CUT_SEARCH_S = 6.0        # im letzten Stück des Fensters nach der leisesten Stelle suchen
MAX_TOKENS_PER_WINDOW = 224


def _split_on_silence(audio: np.ndarray) -> list[np.ndarray]:
    """Teilt in Stücke <= 30 s; Schnitt jeweils an der leisesten 100-ms-Stelle
    in den letzten CUT_SEARCH_S Sekunden des Fensters."""
    win = int(MAX_WINDOW_S * SR) - SR // 2   # etwas Luft, damit EOT sicher erkannt wird
    search = int(CUT_SEARCH_S * SR)
    frame = SR // 10
    chunks, pos = [], 0
    while len(audio) - pos > win:
        lo, hi = pos + win - search, pos + win
        seg = audio[lo:hi]
        n = len(seg) // frame
        energy = (seg[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
        cut = lo + int(np.argmin(energy)) * frame + frame // 2
        chunks.append(audio[pos:cut])
        pos = cut
    if len(audio) - pos > SR // 5:          # Reste unter 0,2 s weglassen
        chunks.append(audio[pos:])
    return chunks


def _to_native(src: np.ndarray, dst: np.ndarray, attr) -> None:
    """Encoder-Ausgabe (NCHW/plain) in das native Eingabe-Layout des Decoders schreiben."""
    if attr.fmt == 2:  # NC1HWC2; bei C <= C2 (hier 8 Köpfe) ist C1 = 1 -> (H, W, C)
        dst.reshape(src.shape[2], src.shape[3], src.shape[1])[...] = src[0].transpose(1, 2, 0)
    else:
        dst.reshape(src.shape)[...] = src


class WhisperRKNN:
    """Whisper Inferenz auf RK3588 NPU (Encoder + Decoder mit KV-Cache, Zero-Copy)."""

    def __init__(self, encoder_path: str, decoder_path: str, model_size: str = "base"):
        self.model_size = model_size
        log.info(f"Lade RKNN Encoder: {encoder_path}")
        self.encoder = rknn_zc.RknnModel(encoder_path, rknn_zc.NPU_CORE_0_1_2)
        log.info(f"Lade RKNN Decoder: {decoder_path}")
        self.decoder = rknn_zc.RknnModel(decoder_path, rknn_zc.NPU_CORE_0)

        def index(attrs, pattern):
            return {re.search(pattern, a.name.decode()).group(1): i
                    for i, a in enumerate(attrs) if re.search(pattern, a.name.decode())}

        d_in, d_out = self.decoder.in_attrs, self.decoder.out_attrs
        self.n_layer = len(index(d_in, r"self_k_(\d+)$"))
        self.self_in = [index(d_in, rf"self_{kv}_(\d+)$")[str(i)]
                        for i in range(self.n_layer) for kv in "kv"]
        self.self_out = [index(d_out, rf"this_self_{kv}_(\d+)$")[str(i)]
                         for i in range(self.n_layer) for kv in "kv"]
        self.cross_in = [index(d_in, rf"cross_{kv}_(\d+)$")[str(i)]
                         for i in range(self.n_layer) for kv in "kv"]
        self.cross_out = [index(self.encoder.out_attrs, rf"cross_{kv}_(\d+)$")[str(i)]
                          for i in range(self.n_layer) for kv in "kv"]
        names = [a.name.decode() for a in d_in]
        self.i_token = next(i for i, n in enumerate(names) if n.endswith("tokens"))
        self.i_offset = next(i for i, n in enumerate(names) if n.endswith("offset"))
        self.i_mask = next(i for i, n in enumerate(names) if n.endswith("mask"))
        self.o_logits = next(i for i, a in enumerate(d_out) if a.name.decode().endswith("logits"))
        self._dirty = None    # None = alle Eingaben beim nächsten Lauf synchronisieren

        tok = whisper.get_tokenizer(multilingual=True, language="en")
        self.sot = tok.sot_sequence_including_notimestamps[0]
        self.transcribe_tok = tok.sot_sequence_including_notimestamps[2]
        self.no_timestamps = tok.sot_sequence_including_notimestamps[3]
        self.eot = tok.eot
        self.timestamp_begin = tok.timestamp_begin
        self.lang_codes = list(whisper.LANGUAGES.keys())[: whisper.NUM_LANGUAGES]
        self.tokenizer = tok
        self.last_language = None
        log.info(f"✅ RKNN Modell bereit (whisper-{model_size}, {self.n_layer} Decoder-Schichten)")

    # ── ein Decoder-Schritt ───────────────────────────────────────────────────
    def _step(self, token: int, offset: int) -> np.ndarray:
        d = self.decoder
        d.inputs[self.i_token].flat[0] = token
        d.inputs[self.i_offset].flat[0] = offset
        mask = d.inputs[self.i_mask]
        mask[:] = 1
        mask[:offset] = 0
        dirty = None
        if self._dirty is not None:
            dirty = [self.i_token, self.i_offset, self.i_mask] + self._dirty
        d.run(dirty)
        for i_in, i_out in zip(self.self_in, self.self_out):   # neue Self-K/V-Zeile in den Cache
            d.inputs[i_in][0, offset, :] = d.outputs[i_out].reshape(-1)
        self._dirty = self.self_in
        return d.outputs[self.o_logits].reshape(-1).astype(np.float32)

    def _window(self, chunk: np.ndarray, language: str | None) -> tuple[str, str]:
        e, d = self.encoder, self.decoder
        e.inputs[0][0] = whisper.log_mel_spectrogram(whisper.pad_or_trim(chunk))
        e.run()
        for i_out, i_in in zip(self.cross_out, self.cross_in):   # einmal pro Fenster
            _to_native(e.outputs[i_out], d.inputs[i_in], d.in_attrs[i_in])
        for i in self.self_in:
            d.inputs[i][...] = 0
        self._dirty = None

        logits = self._step(self.sot, 0)
        if language is None:
            lang_ids = np.arange(self.sot + 1, self.sot + 1 + len(self.lang_codes))
            language = self.lang_codes[int(np.argmax(logits[lang_ids]))]
        lang_tok = self.sot + 1 + self.lang_codes.index(language)

        offset = 1
        for t in (lang_tok, self.transcribe_tok, self.no_timestamps):
            logits = self._step(t, offset)
            offset += 1

        tokens: list[int] = []
        while offset < N_TEXT_CTX and len(tokens) < MAX_TOKENS_PER_WINDOW:
            logits[self.timestamp_begin:] = -np.inf          # keine Zeitstempel
            logits[self.sot:self.timestamp_begin] = -np.inf   # keine Sondertokens außer EOT
            nxt = int(np.argmax(logits))
            if nxt == self.eot:
                break
            tokens.append(nxt)
            # Endlosschleife abfangen (gleicher Token 12x in Folge)
            if len(tokens) >= 12 and len(set(tokens[-12:])) == 1:
                tokens = tokens[:-11]
                break
            logits = self._step(nxt, offset)
            offset += 1
        return self.tokenizer.decode(tokens).strip(), language

    def transcribe(self, wav_path: str, language: str = "auto") -> str:
        """
        WAV-Datei → Transkript via RKNN NPU.
        language: ISO-639-1 Code (z.B. 'de', 'en') oder 'auto' (Erkennung im ersten Fenster).
        """
        t0 = time.monotonic()
        audio = whisper.load_audio(wav_path)
        lang = None if language in (None, "", "auto") else language.lower()
        if lang is not None and lang not in self.lang_codes:
            raise ValueError(f"Unbekannte Sprache: {language}")

        parts = []
        for chunk in _split_on_silence(audio):
            text, lang = self._window(chunk, lang)   # erkannte Sprache für alle weiteren Fenster
            if text:
                parts.append(text)
        transcript = " ".join(parts)
        self.last_language = lang
        log.info(f"RKNN Transkription: {len(audio) / SR:.0f}s Audio, Sprache={lang}, "
                 f"{len(transcript)} Zeichen in {time.monotonic() - t0:.1f}s")
        return transcript

    def release(self):
        self.encoder.release()
        self.decoder.release()
        log.info("RKNN Modell freigegeben")
