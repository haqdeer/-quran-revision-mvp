# app.py
# Step 10.5 — Token repair + training log + reliable copy + safe resample (simple + future-ready)

import os
import re
import json
import time
import math
import datetime
from difflib import SequenceMatcher

import numpy as np
import torch
import torchaudio
import gradio as gr
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# -----------------------------
# Config
# -----------------------------
MODEL_ID = os.getenv("ASR_MODEL_ID", "elgeish/wav2vec2-large-xlsr-53-arabic")
TARGET_SR = 16000
LOG_PATH = "training_log.jsonl"

# Similarity thresholds (MVP)
SIM_OK = 0.55          # if >= this, we treat as acceptable match
SIM_STRICT = 0.72      # used for "confident" in UI text

# Tokens shorter than this are often ASR fragments; we will try to REPAIR them instead of trusting them.
MIN_TOKEN_LEN = 4

# For silence/stuck detection
SILENCE_DB_THRESHOLD = -38.0  # approx
STUCK_TAIL_SECONDS = 1.2      # you paused ~1.3–4.8s many times; so keep it modest

# -----------------------------
# Quran (MVP: Surah Al-Fatiha only)
# You can extend later without changing core logic.
# -----------------------------
QURAN = {
    (1, 1): "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    (1, 2): "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
    (1, 3): "الرَّحْمَٰنِ الرَّحِيمِ",
    (1, 4): "مَالِكِ يَوْمِ الدِّينِ",
    (1, 5): "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
    (1, 6): "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
    (1, 7): "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
}

def list_ayat():
    items = []
    for (s, a) in sorted(QURAN.keys()):
        items.append(f"{s}:{a} — {QURAN[(s,a)]}")
    return items

def parse_choice(choice: str):
    # "1:2 — ...."
    m = re.match(r"^\s*(\d+)\s*:\s*(\d+)", choice)
    if not m:
        return (1, 1)
    return (int(m.group(1)), int(m.group(2)))

# -----------------------------
# Text helpers
# -----------------------------
AR_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

def strip_diacritics(ar: str) -> str:
    ar = AR_DIACRITICS.sub("", ar)
    # normalize common letters
    ar = ar.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    ar = ar.replace("ى","ي").replace("ؤ","و").replace("ئ","ي")
    ar = ar.replace("ة","ه")
    ar = re.sub(r"\s+", " ", ar).strip()
    return ar

def arabic_words_no_diacritics(ar_with_harakat: str):
    s = strip_diacritics(ar_with_harakat)
    # keep Arabic letters only + spaces
    s = re.sub(r"[^\u0600-\u06FF\s]", " ", s)
    words = [w for w in s.split() if w]
    return words

# Arabic -> simple Latin proxy
# IMPORTANT: we do NOT use str.maketrans (it crashes on multi-length keys).
def ar_to_proxy(ar: str) -> str:
    ar = strip_diacritics(ar)
    # remove tatweel
    ar = ar.replace("ـ", "")
    # very light mapping
    m = {
        "ا":"a","ب":"b","ت":"t","ث":"th","ج":"j","ح":"h","خ":"kh",
        "د":"d","ذ":"dh","ر":"r","ز":"z","س":"s","ش":"sh","ص":"s",
        "ض":"d","ط":"t","ظ":"z","ع":"a","غ":"gh","ف":"f","ق":"q",
        "ك":"k","ل":"l","م":"m","ن":"n","ه":"h","و":"w","ي":"y",
        "ء":"", " " : " ",
        "ة":"h",
    }
    out = []
    for ch in ar:
        out.append(m.get(ch, ""))
    s = "".join(out)
    s = re.sub(r"\s+", " ", s).strip()
    # compress doubles
    s = re.sub(r"(.)\1+", r"\1", s)
    return s

def normalize_asr_text(t: str) -> str:
    t = t.lower()
    # remove weird punctuation
    t = re.sub(r"[^a-z\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokenize_asr(raw: str):
    raw_n = normalize_asr_text(raw)
    toks = [t for t in raw_n.split() if t]
    return toks

def sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()

# -----------------------------
# Audio helpers
# -----------------------------
def to_float32_wave(y: np.ndarray) -> np.ndarray:
    # y can be int16 from mic
    if y is None:
        return None
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype == np.int32:
        y = y.astype(np.float32) / 2147483648.0
    else:
        y = y.astype(np.float32)
    # if stereo -> mono
    if y.ndim == 2 and y.shape[1] > 1:
        y = y.mean(axis=1)
    return y

def resample_to_target(y: np.ndarray, sr: int) -> np.ndarray:
    y = to_float32_wave(y)
    if y is None:
        return None
    if sr == TARGET_SR:
        return y
    wav = torch.tensor(y, dtype=torch.float32)
    out = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return out.numpy()

def tail_silence_seconds(y: np.ndarray, sr: int) -> float:
    # rough dB threshold on amplitude
    if y is None or len(y) == 0:
        return 0.0
    y = to_float32_wave(y)
    eps = 1e-9
    # amplitude threshold from dB
    thr = 10 ** (SILENCE_DB_THRESHOLD / 20.0)  # convert to linear
    idx = np.where(np.abs(y) > thr)[0]
    if len(idx) == 0:
        return len(y) / sr
    last = idx[-1]
    tail = (len(y) - 1 - last) / sr
    return float(max(0.0, tail))

# -----------------------------
# Load model
# -----------------------------
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()

@torch.inference_mode()
def asr_transcribe(y: np.ndarray, sr: int) -> str:
    y16 = resample_to_target(y, sr)
    inputs = processor(y16, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    logits = model(**inputs).logits
    pred_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(pred_ids)[0]
    return text

# -----------------------------
# Token repair logic (merge/split tolerant)
# -----------------------------
def repair_tokens(tokens, expected_proxies):
    """
    Attempt to fix common ASR fragmentation:
    - keep long tokens
    - for short tokens (< MIN_TOKEN_LEN), try joining with neighbor(s) if it improves match to any expected proxy
    """
    if not tokens:
        return []

    repaired = []
    i = 0
    while i < len(tokens):
        t = tokens[i]

        # if token is short, try joining with next or prev
        if len(t) < MIN_TOKEN_LEN:
            candidates = [t]
            if repaired:
                candidates.append(repaired[-1] + t)  # join with prev
            if i + 1 < len(tokens):
                candidates.append(t + tokens[i+1])  # join with next
                if repaired:
                    candidates.append(repaired[-1] + t + tokens[i+1])  # join prev+short+next

            # score candidates by best similarity to any expected proxy
            def best_score(x):
                return max((sim(x, e) for e in expected_proxies), default=0.0)

            best = max(candidates, key=best_score)
            # If best uses next token (t+next), we consume next
            used_next = (i + 1 < len(tokens)) and (best.endswith(tokens[i+1])) and (best != t) and (best != (repaired[-1] + t if repaired else ""))
            used_prev = repaired and (best.startswith(repaired[-1])) and (best != t)

            # apply replacement carefully
            if used_prev:
                repaired[-1] = best  # replace prev with merged
            else:
                repaired.append(best)

            if used_next and i + 1 < len(tokens):
                i += 2
            else:
                i += 1
            continue

        # normal token
        repaired.append(t)
        i += 1

    # remove duplicates caused by merges (simple)
    out = []
    for t in repaired:
        if not out or out[-1] != t:
            out.append(t)
    return out

# -----------------------------
# Alignment (simple greedy)
# -----------------------------
def align_expected_to_tokens(expected_words, expected_proxies, tokens):
    """
    Greedy matching:
    for each expected word, pick the best remaining token (in order)
    """
    alignment = []
    used = [False] * len(tokens)

    for ew, ep in zip(expected_words, expected_proxies):
        best_j = None
        best_s = 0.0
        for j, tok in enumerate(tokens):
            if used[j]:
                continue
            s = sim(tok, ep)
            if s > best_s:
                best_s = s
                best_j = j
        if best_j is None or best_s < SIM_OK:
            alignment.append((ew, ep, None, 0.0, "MISS"))
        else:
            used[best_j] = True
            status = "OK" if best_s >= SIM_STRICT else "UNCERTAIN"
            alignment.append((ew, ep, tokens[best_j], best_s, status))

    extras = [tokens[i] for i, u in enumerate(used) if not u]
    return alignment, extras

def summarize_alignment(alignment):
    missing = []
    wrong = []
    matched_ok = 0
    for ew, ep, tok, s, status in alignment:
        if tok is None:
            missing.append(ew)
        else:
            if status == "OK":
                matched_ok += 1
            else:
                # uncertain counts as "possible wrong"
                wrong.append((ew, tok, s))
    return matched_ok, missing, wrong

def format_alignment_lines(alignment):
    lines = []
    for ew, ep, tok, s, status in alignment:
        if tok is None:
            lines.append(f"- {ew}  ->  ❌ (no token aligned)")
        else:
            icon = "✅" if status == "OK" else "⚠️"
            lines.append(f"- {ew}  ->  {icon} {tok}  (sim {s:.2f})")
    return "\n".join(lines)

# -----------------------------
# Training log
# -----------------------------
def append_log(entry: dict):
    entry["ts"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def ensure_log_exists():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            pass

def download_log_file():
    ensure_log_exists()
    return LOG_PATH

# -----------------------------
# Main runner
# -----------------------------
def run_check(choice, audio):
    """
    audio from gr.Audio(type="numpy") returns (sr, np.array)
    """
    if audio is None:
        return "⚠️ Please record audio first.", ""

    surah, ayah = parse_choice(choice)
    expected = QURAN.get((surah, ayah), "")
    if not expected:
        return "⚠️ This ayah is not in the MVP dataset yet.", ""

    sr, y = audio
    y = np.array(y)
    dur = len(y) / float(sr) if sr else 0.0
    tail = tail_silence_seconds(y, sr)

    # ASR
    raw = asr_transcribe(y, sr)

    # Expected words + proxies
    expected_words = arabic_words_no_diacritics(expected)
    expected_proxies = [ar_to_proxy(w) for w in expected_words]

    # Tokens
    tokens = tokenize_asr(raw)
    repaired = repair_tokens(tokens, expected_proxies)

    # Align
    alignment, extras = align_expected_to_tokens(expected_words, expected_proxies, repaired)
    matched_ok, missing, wrong = summarize_alignment(alignment)

    # "next expected hint" = first missing word, else last word
    hint = missing[0] if missing else expected_words[-1] if expected_words else ""

    # stuck detection
    stuck = tail >= STUCK_TAIL_SECONDS

    # Build output text
    out = []
    out.append(f"✅ Start point: {surah}:{ayah}")
    out.append(f"Expected (with harakaat): {expected}")
    out.append(f"Expected words: {expected_words}")
    out.append(f"Expected proxy: {expected_proxies}")
    out.append("")
    out.append(f"🎙️ Duration: {dur:.2f}s | SR: {sr}")
    out.append(f"📝 ASR Raw: {raw}")
    out.append(f"🧩 ASR Tokens (cleaned): {repaired}")
    out.append("")
    out.append("---")
    out.append(f"Matched (OK): {matched_ok}/{len(expected_words)} | ASR tokens: {len(repaired)}")

    if missing:
        out.append(f"❌ Missing word(s): {', '.join(missing)}")
    if wrong:
        wtxt = " , ".join([f"{w[0]}⇢{w[1]}({w[2]:.2f})" for w in wrong[:4]])
        out.append(f"⚠️ Wrong/uncertain word(s): {wtxt}")
    if extras:
        out.append(f"⚠️ Extra token(s): {', '.join(extras[:6])}")

    out.append(f"➡️ Next expected word (hint): **{hint}**")
    out.append("")
    out.append("🔍 Per-word alignment:")
    out.append(format_alignment_lines(alignment))

    if stuck:
        out.append("")
        out.append(f"🧠 Possible stuck detected (you paused ~{tail:.1f}s at the end).")
        out.append(f"Hint: next word is **{hint}**")

    out.append("")
    out.append("Rule: Select ONE ayah and recite ONLY that ayah.")
    out.append("Next step after this: we’ll keep building your training log (no heavy model training).")

    result_text = "\n".join(out)

    # Save training log
    entry = {
        "surah": surah,
        "ayah": ayah,
        "expected": expected,
        "expected_words": expected_words,
        "expected_proxies": expected_proxies,
        "asr_raw": raw,
        "tokens_before": tokens,
        "tokens_repaired": repaired,
        "alignment": [
            {"expected": ew, "proxy": ep, "token": tok, "sim": float(s), "status": st}
            for (ew, ep, tok, s, st) in alignment
        ],
        "missing": missing,
        "wrong": [{"expected": w0, "token": w1, "sim": float(w2)} for (w0, w1, w2) in wrong],
        "extras": extras,
        "duration_sec": float(dur),
        "sr": int(sr),
        "tail_silence_sec": float(tail),
        "stuck": bool(stuck),
    }
    append_log(entry)

    # Also return a compact “one-line” for quick copy if you want later
    compact = f"{surah}:{ayah} | ASR: {raw} | missing: {','.join(missing) if missing else '-'}"
    return result_text, compact

# -----------------------------
# UI
# -----------------------------
with gr.Blocks(title="Qur’an Recitation Checker (MVP)") as demo:
    gr.Markdown("## Qur’an Recitation Checker (MVP)\n"
                "Pick **one ayah**, recite **only that ayah**, then click **Check**.\n\n"
                "This version saves your attempts to a **training_log.jsonl** file automatically.")

    with gr.Row():
        ayah_choice = gr.Dropdown(choices=list_ayat(), value=list_ayat()[0], label="Start point (Surah:Ayah)")
    audio = gr.Audio(sources=["microphone"], type="numpy", label="Record (mic)")

    with gr.Row():
        btn = gr.Button("✅ Check Recitation", variant="primary")
        btn_dl = gr.Button("⬇️ Download Training Log")
    out_text = gr.Textbox(label="Result", lines=22)
    compact_text = gr.Textbox(label="Compact (optional)", lines=1)

    with gr.Row():
        btn_copy = gr.Button("📋 Copy Result")
        copy_status = gr.Textbox(label="Copy status", lines=1)

    log_file = gr.File(label="training_log.jsonl", visible=True)

    btn.click(fn=run_check, inputs=[ayah_choice, audio], outputs=[out_text, compact_text])

    # Download log
    btn_dl.click(fn=download_log_file, inputs=None, outputs=log_file)

    # Copy (NO undefined anymore)
    # We do copy on the client side and return a status string.
    btn_copy.click(
        fn=lambda txt: txt,
        inputs=out_text,
        outputs=copy_status,
        js="""
        (txt) => {
            try {
                navigator.clipboard.writeText(txt || "");
                return "Copied ✅";
            } catch (e) {
                return "Copy failed ❌ (browser blocked clipboard)";
            }
        }
        """
    )

demo.queue().launch()
