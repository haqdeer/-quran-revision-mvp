import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import gradio as gr
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# -----------------------------
# CONFIG
# -----------------------------
MODEL_ID = "elgeish/wav2vec2-large-xlsr-53-arabic"
TARGET_SR = 16000

# "Silent listener" tuning
MIN_TOKEN_LEN = 3              # ignore tokens shorter than this (e.g., "al", "r")
OK_SIM_THRESHOLD = 0.62        # similarity needed to treat a token as "good enough"
LOWCONF_THRESHOLD = 0.45       # below this = low confidence
TWO_STRIKE = 2                 # only warn after same issue repeats
MAX_OUTPUT_LINES = 40          # keep UI clean

# -----------------------------
# Quran test set (for now)
# Extend later using dataset, but MVP = fixed
# -----------------------------
QURAN = {
    "1:1": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    "1:2": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
    "1:3": "الرَّحْمَٰنِ الرَّحِيمِ",
    "1:7": "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
}

# -----------------------------
# Helpers: Arabic normalization
# -----------------------------
AR_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
AR_PUNCT = re.compile(r"[^\u0600-\u06FF\s]")

def strip_diacritics(s: str) -> str:
    return AR_DIACRITICS.sub("", s)

def normalize_ar(s: str) -> str:
    s = strip_diacritics(s)
    s = AR_PUNCT.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def arabic_words(s: str) -> List[str]:
    s = normalize_ar(s)
    return [w for w in s.split(" ") if w]

# -----------------------------
# Proxy: convert Arabic word -> simple latin-ish skeleton
# NOTE: We do NOT use str.translate with multi-char keys (your earlier crash).
# We keep it simple and safe.
# -----------------------------
AR2LAT_SINGLE = {
    "ا":"a","أ":"a","إ":"i","آ":"a","ء":"", "ؤ":"w","ئ":"y",
    "ب":"b","ت":"t","ث":"th","ج":"j","ح":"h","خ":"kh",
    "د":"d","ذ":"dh","ر":"r","ز":"z","س":"s","ش":"sh",
    "ص":"s","ض":"d","ط":"t","ظ":"z","ع":"a","غ":"gh",
    "ف":"f","ق":"q","ك":"k","ل":"l","م":"m","ن":"n",
    "ه":"h","و":"w","ي":"y","ى":"a","ة":"h",
    " ":" ",
    "َ":"", "ُ":"", "ِ":"", "ْ":"", "ّ":"", "ٰ":""
}

def ar_to_proxy(word: str) -> str:
    word = strip_diacritics(word)
    out = []
    for ch in word:
        out.append(AR2LAT_SINGLE.get(ch, ""))
    proxy = "".join(out)
    proxy = proxy.lower()
    proxy = re.sub(r"[^a-z]+", "", proxy)
    # remove common leading article "al" noise impact a bit
    proxy = re.sub(r"^al+", "al", proxy)
    return proxy

# -----------------------------
# Similarity (simple + fast)
# -----------------------------
def bigrams(s: str) -> set:
    if len(s) < 2:
        return set()
    return {s[i:i+2] for i in range(len(s)-1)}

def sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

# -----------------------------
# Audio processing
# -----------------------------
def to_float32_audio(y: np.ndarray) -> np.ndarray:
    # Gradio often returns int16 from mic
    if y is None:
        return y
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype != np.float32:
        y = y.astype(np.float32)
    # mono
    if y.ndim == 2:
        y = np.mean(y, axis=1).astype(np.float32)
    return y

def resample_audio(y: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SR:
        return y
    y_t = torch.tensor(y, dtype=torch.float32)
    y_rs = torchaudio.functional.resample(y_t, sr, TARGET_SR)
    return y_rs.numpy()

# -----------------------------
# ASR
# -----------------------------
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()

def run_asr(y: np.ndarray, sr: int) -> str:
    y = to_float32_audio(y)
    y = resample_audio(y, sr)

    inputs = processor(y, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    # keep only letters/spaces, lower
    text = text.strip()
    return text

def clean_tokens(asr_text: str) -> List[str]:
    t = asr_text.lower()
    t = re.sub(r"[^a-z\s~_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    toks = [x for x in t.split(" ") if x]
    # remove ultra-short tokens that create false alarms
    toks = [x for x in toks if len(x) >= MIN_TOKEN_LEN]
    return toks

# -----------------------------
# Alignment + "silent listener" decision
# -----------------------------
@dataclass
class CheckResult:
    status: str                # "OK", "LOWCONF", "WARN"
    note: str                  # human-friendly
    details: str               # optional debug text (limited)

def align_expected_to_tokens(expected_words: List[str], tokens: List[str]) -> Tuple[List[Tuple[str, Optional[str], float]], float]:
    """
    Greedy alignment: for each expected word proxy, pick best token similarity.
    Returns per-word alignment list and average similarity.
    """
    proxies = [ar_to_proxy(w) for w in expected_words]
    used = set()
    aligned = []
    sims = []

    for w, p in zip(expected_words, proxies):
        best_j, best_s = None, 0.0
        for j, tok in enumerate(tokens):
            if j in used:
                continue
            s = sim(p, tok)
            if s > best_s:
                best_s = s
                best_j = j
        if best_j is not None and best_s >= LOWCONF_THRESHOLD:
            used.add(best_j)
            aligned.append((w, tokens[best_j], best_s))
            sims.append(best_s)
        else:
            aligned.append((w, None, 0.0))
            sims.append(0.0)

    avg = float(np.mean(sims)) if sims else 0.0
    return aligned, avg

def decide_silent_listener(expected_words: List[str], tokens: List[str], strike_state: Dict[str, Dict[str, int]], key: str) -> CheckResult:
    aligned, avg = align_expected_to_tokens(expected_words, tokens)

    missing = [w for (w, tok, s) in aligned if tok is None]
    weak = [(w, tok, s) for (w, tok, s) in aligned if tok is not None and s < OK_SIM_THRESHOLD]

    # If ASR is too messy, do NOT blame user
    if avg < LOWCONF_THRESHOLD and len(tokens) <= 3:
        return CheckResult(
            status="LOWCONF",
            note="🟡 Low confidence (ASR noisy). Try again (no judgment).",
            details=""
        )

    # Silent OK if nothing missing and weak is small
    if not missing and len(weak) <= 1:
        # reset strikes for this key
        strike_state[key] = {}
        return CheckResult(
            status="OK",
            note="✅ Looks OK (silent listener).",
            details=""
        )

    # If missing exists, apply two-strike logic
    # Only warn if the SAME missing word repeats
    if missing:
        first_miss = missing[0]
        strike_state.setdefault(key, {})
        strike_state[key][first_miss] = strike_state[key].get(first_miss, 0) + 1

        if strike_state[key][first_miss] >= TWO_STRIKE:
            return CheckResult(
                status="WARN",
                note=f"🛑 Possible skip near: **{first_miss}** (repeated {TWO_STRIKE}x)",
                details=_format_alignment(aligned)
            )
        else:
            return CheckResult(
                status="LOWCONF",
                note=f"🟡 Maybe check: **{first_miss}** (I’ll only stop if it repeats).",
                details=""
            )

    # If no missing but many weak, treat as low confidence
    return CheckResult(
        status="LOWCONF",
        note="🟡 I heard it, but confidence is low. Please try once more.",
        details=""
    )

def _format_alignment(aligned: List[Tuple[str, Optional[str], float]]) -> str:
    lines = ["🔍 Per-word (approx):"]
    for w, tok, s in aligned:
        if tok is None:
            lines.append(f"- {w} -> ❌")
        else:
            lines.append(f"- {w} -> {tok} ({s:.2f})")
    return "\n".join(lines[:MAX_OUTPUT_LINES])

# -----------------------------
# Gradio App
# -----------------------------
def run_check(ayah_key: str, audio: Tuple[int, np.ndarray], state: dict):
    if state is None:
        state = {}
    strike_state = state.setdefault("strikes", {})

    if ayah_key not in QURAN:
        return "Choose a valid start point.", state

    sr, y = audio
    if y is None:
        return "Please record audio.", state

    expected = QURAN[ayah_key]
    exp_words = arabic_words(expected)

    t0 = time.time()
    asr_raw = run_asr(y, sr)
    toks = clean_tokens(asr_raw)
    dt = time.time() - t0

    result = decide_silent_listener(exp_words, toks, strike_state, ayah_key)

    # Output: short + human-like
    out_lines = []
    out_lines.append(f"Start: {ayah_key}")
    out_lines.append(f"Expected: {expected}")
    out_lines.append("")
    out_lines.append(f"Duration: {len(y)/sr:.2f}s | SR: {sr}")
    out_lines.append(f"ASR: {asr_raw}")
    out_lines.append("")
    out_lines.append(result.note)

    # only show details when we actually stop (WARN)
    if result.status == "WARN" and result.details:
        out_lines.append("")
        out_lines.append(result.details)

    return "\n".join(out_lines), state

with gr.Blocks() as demo:
    gr.Markdown("## Qur’an Revision (Silent Listener Mode v1)")
    gr.Markdown("This is an MVP. It will **stay quiet** unless an issue repeats.")
    with gr.Row():
        ayah = gr.Dropdown(choices=list(QURAN.keys()), value="1:1", label="Start point")
    audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record / Upload")
    state = gr.State({})
    btn = gr.Button("Check recitation")
    out = gr.Textbox(label="Result", lines=18)
    btn.click(run_check, inputs=[ayah, audio, state], outputs=[out, state])

demo.queue().launch()