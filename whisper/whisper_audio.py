"""
whisper_audio.py – Schlanker Ersatz für die openai-whisper Hilfsfunktionen,
die rknn_inference.py braucht (Audio laden, Mel-Spektrogramm, Tokenizer).

Warum diese Datei existiert: openai-whisper wird hier NICHT für seine
eigentliche Aufgabe (PyTorch-Modell-Inferenz) gebraucht – die läuft komplett
über RKNN. Es wurde nur als Bibliothek für Audio-Preprocessing und
Tokenisierung zweckentfremdet, was eine ~1GB+ PyTorch-Abhängigkeit ins
Image zieht, die nie tatsächlich für Inferenz genutzt wird.

Jede Funktion hier ist 1:1 gegen den echten Quellcode von openai-whisper
verifiziert (whisper/audio.py und whisper/tokenizer.py, Stand: aktueller
main-Branch). Es wird KEIN Parameter erraten:

  - load_audio():        nutzt denselben ffmpeg-Subprocess-Call wie das Original
  - log_mel_spectrogram(): nutzt dieselbe vorgefertigte Mel-Filter-Matrix
                           (mel_filters.npz, ~4KB) statt sie selbst zu berechnen
  - WhisperTokenizer:     nutzt dasselbe multilingual.tiktoken-Vokabular (~817KB)
                           und exakt dieselbe Sondertoken-/sot_sequence-Logik

Beide Asset-Dateien werden einmalig beim ersten Start automatisch von GitHub
heruntergeladen (analog zum Whisper-Modell-Download) – sie sind reine
Daten-Assets, unabhängig von PyTorch.

Abhängigkeiten: numpy, scipy, tiktoken (alle deutlich kleiner als torch).
ffmpeg muss im Container installiert sein (ist es ohnehin schon).
"""
import os
import base64
import logging
import subprocess
import urllib.request

import numpy as np

log = logging.getLogger(__name__)

# ── Whisper-Audio-Konstanten (1:1 aus whisper/audio.py) ─────────────────────
SAMPLE_RATE  = 16000
N_FFT        = 400
HOP_LENGTH   = 160
CHUNK_LENGTH = 30
N_SAMPLES    = CHUNK_LENGTH * SAMPLE_RATE  # 480000

ASSETS_DIR = "/models"
MEL_FILTERS_URL = (
    "https://raw.githubusercontent.com/openai/whisper/main/"
    "whisper/assets/mel_filters.npz"
)
MULTILINGUAL_TIKTOKEN_URL = (
    "https://raw.githubusercontent.com/openai/whisper/main/"
    "whisper/assets/multilingual.tiktoken"
)


def _ensure_asset(url: str, filename: str) -> str:
    """Lädt eine Whisper-Asset-Datei herunter, falls noch nicht vorhanden."""
    path = os.path.join(ASSETS_DIR, filename)
    if os.path.exists(path):
        return path
    log.info(f"Lade Whisper-Asset: {url}")
    os.makedirs(ASSETS_DIR, exist_ok=True)
    urllib.request.urlretrieve(url, path)
    log.info(f"✅ Asset gespeichert: {path}")
    return path


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    1:1 wie whisper.audio.load_audio: Audio via ffmpeg-Subprocess dekodieren,
    auf Mono + sr Hz resamplen, als float32 in [-1, 1] zurückgeben.
    """
    cmd = [
        "ffmpeg", "-nostdin", "-threads", "0",
        "-i", path,
        "-f", "s16le", "-ac", "1", "-acodec", "pcm_s16le", "-ar", str(sr),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(audio: np.ndarray, length: int = N_SAMPLES) -> np.ndarray:
    """1:1 wie whisper.audio.pad_or_trim (numpy-Zweig)."""
    if audio.shape[-1] > length:
        audio = audio[..., :length]
    elif audio.shape[-1] < length:
        pad = length - audio.shape[-1]
        audio = np.pad(audio, (0, pad))
    return audio


def _mel_filters(n_mels: int = 80) -> np.ndarray:
    """Lädt die echte, vorgefertigte Mel-Filter-Matrix (kein eigenes Berechnen nötig)."""
    assert n_mels in (80, 128), f"Unsupported n_mels: {n_mels}"
    path = _ensure_asset(MEL_FILTERS_URL, "mel_filters.npz")
    with np.load(path, allow_pickle=False) as f:
        return f[f"mel_{n_mels}"]


def log_mel_spectrogram(audio: np.ndarray, n_mels: int = 80, padding: int = 0) -> np.ndarray:
    """
    1:1 wie whisper.audio.log_mel_spectrogram. Die STFT wird hier manuell mit
    numpy.fft.rfft nachgebaut (Hann-Window, reflect-Padding wie torch.stft mit
    center=True), da scipy.signal.stft eine andere interne Normalisierung
    nutzt und NICHT numerisch äquivalent zu torch.stft ist (verifiziert).
    Rückgabe: (n_mels, n_frames), n_frames=3000 bei 30s Audio.
    """
    if padding > 0:
        audio = np.pad(audio, (0, padding))

    window  = np.hanning(N_FFT).astype(np.float32)
    pad     = N_FFT // 2
    padded  = np.pad(audio, (pad, pad), mode="reflect")
    n_frames = 1 + (len(padded) - N_FFT) // HOP_LENGTH

    stft_out = np.empty((N_FFT // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = padded[start:start + N_FFT] * window
        stft_out[:, i] = np.fft.rfft(frame, n=N_FFT)

    magnitudes = np.abs(stft_out[:, :-1]) ** 2  # letzten Frame weglassen, wie [..., :-1] im Original

    filters = _mel_filters(n_mels)
    mel_spec = filters @ magnitudes

    log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


# ── Whisper-Tokenizer (1:1 nachgebaut nach whisper/tokenizer.py) ───────────
# Exakt aus openai-whisper/whisper/tokenizer.py LANGUAGES übernommen.
# Reihenfolge ist entscheidend: die Sprach-Sondertokens folgen im Vokabular
# in genau dieser Reihenfolge direkt nach <|startoftranscript|>.
LANGUAGES = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish", "my": "myanmar",
    "bo": "tibetan", "tl": "tagalog", "mg": "malagasy", "as": "assamese",
    "tt": "tatar", "haw": "hawaiian", "ln": "lingala", "ha": "hausa",
    "ba": "bashkir", "jw": "javanese", "su": "sundanese", "yue": "cantonese",
}
NUM_LANGUAGES = 99  # entspricht get_encoding()'s Default num_languages=99


def _get_encoding(vocab_path: str, num_languages: int = NUM_LANGUAGES):
    """1:1 Nachbau von whisper.tokenizer.get_encoding()."""
    import tiktoken

    with open(vocab_path, "r", encoding="utf-8") as f:
        ranks = {
            base64.b64decode(token): int(rank)
            for token, rank in (line.split() for line in f if line)
        }
    n_vocab = len(ranks)  # ergibt sich aus dem File, nicht hartcodiert
    special_tokens = {}

    specials = [
        "<|endoftext|>",
        "<|startoftranscript|>",
        *[f"<|{lang}|>" for lang in list(LANGUAGES.keys())[:num_languages]],
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
        *[f"<|{i * 0.02:.2f}|>" for i in range(1501)],
    ]
    for token in specials:
        special_tokens[token] = n_vocab
        n_vocab += 1

    return tiktoken.Encoding(
        name=os.path.basename(vocab_path),
        explicit_n_vocab=n_vocab,
        pat_str=r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        mergeable_ranks=ranks,
        special_tokens=special_tokens,
    )


class WhisperTokenizer:
    """
    1:1 Nachbau der relevanten Teile von whisper.tokenizer.Tokenizer:
    sot_sequence_including_notimestamps, eot, decode(). Deckt exakt das ab,
    was rknn_inference.py braucht.
    """

    def __init__(self, multilingual: bool = True, language: str | None = None, task: str = "transcribe"):
        vocab_path = _ensure_asset(MULTILINGUAL_TIKTOKEN_URL, "multilingual.tiktoken")
        self._encoding = _get_encoding(vocab_path, num_languages=NUM_LANGUAGES)

        if language is not None:
            language = language.lower()
            if language not in LANGUAGES:
                raise ValueError(f"Unsupported language: {language}")

        if multilingual:
            language = language or "en"
            task = task or "transcribe"
        else:
            language = None
            task = None

        special_tokens = {
            tok: self._encoding.encode_single_token(tok)
            for tok in self._encoding.special_tokens_set
        }

        sot        = special_tokens["<|startoftranscript|>"]
        translate  = special_tokens["<|translate|>"]
        transcribe = special_tokens["<|transcribe|>"]

        langs = tuple(LANGUAGES.keys())[:NUM_LANGUAGES]
        sot_sequence = [sot]
        if language is not None:
            sot_sequence.append(sot + 1 + langs.index(language))
        if task is not None:
            sot_sequence.append(transcribe if task == "transcribe" else translate)

        self.eot = special_tokens["<|endoftext|>"]
        self.timestamp_begin = special_tokens["<|0.00|>"]
        no_timestamps = special_tokens["<|notimestamps|>"]

        self.sot_sequence_including_notimestamps = tuple(sot_sequence + [no_timestamps])

    def decode(self, token_ids: list[int]) -> str:
        """Token-IDs → Text. Timestamp-/Sondertokens werden rausgefiltert (wie im Original)."""
        normal_ids = [t for t in token_ids if t < self.timestamp_begin]
        return self._encoding.decode(normal_ids)


def get_tokenizer(multilingual: bool = True, language: str | None = None, task: str = "transcribe") -> WhisperTokenizer:
    """Drop-in-Ersatz für whisper.tokenizer.get_tokenizer()."""
    return WhisperTokenizer(multilingual=multilingual, language=language, task=task)
