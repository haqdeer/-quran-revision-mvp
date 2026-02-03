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

# Matching
MIN_TOKEN_LEN = 3
OK_SIM_THRESHOLD = 0.68
LOWCONF_THRESHOLD = 0.50

# Stop gating (make it RARE)
STRIKES_TO_STOP = 3
STOP_MIN_AVG_SIM = 0.60
STOP_MAX_WORDS = 6   # for longer ayah, never stop (ASR too noisy)

# -----------------------------
# Quran MVP set
# -----------------------------
QURAN = {
    "1:1": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    "1:2": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
    "1:3": "الرَّحْمَٰنِ الرَّحِيمِ",
    "1:7": "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ",
}

# -----------------------------
# Arabic normalization
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
# Proxy mapping (single-char safe)
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
    proxy = "".join(out).lower()
    proxy = re.sub(r"[^a-z]+", "", proxy)
    proxy = re.sub(r"^al+", "al", proxy)
    return proxy

# -----------------------------
# Similarity (bigrams)
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
# Audio
# -----------------------------
def to_float32_audio(y: np.ndarray) -> np.ndarray:
    if y is None:
        return y
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype != np.float32:
        y = y.astype(np.float32)
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
    return text.strip()

def clean_tokens(asr_text: str) -> List[str]:
    t = asr_text.lower()
    t = re.sub(r"[^a-z\s~_]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    toks = [x for x in t.split(" ") if x]
    toks = [x for x in toks if len(x) >= MIN_TOKEN_LEN]
    return toks

# -----------------------------
# Alignment + decision
# -----------------------------
@dataclass
class CheckResult:
    status: str   # OK / LOW / STOP
    note: str
    avg_sim: float
    first_issue: Optional[str]
    details: str

def align_expected_to_tokens(expected_words: List[str], tokens: List[str]) -> Tuple[List[Tuple[str, Optional[str], float]], float]:
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
                best_s, best_j = s, j

        if best_j is not None and best_s >= LOWCONF_THRESHOLD:
            used.add(best_j)
            aligned.append((w, tokens[best_j], best_s))
            sims.append(best_s)
        else:
            aligned.append((w, None, 0.0))
            sims.append(0.0)

    avg = float(np.mean(sims)) if sims else 0.0
    return aligned, avg

def format_alignment(aligned):
    lines = ["🔍 Per-word (approx):"]
    for w, tok, s in aligned:
        if tok is None:
            lines.append(f"- {w} -> ❌")
        else:
            lines.append(f"- {w} -> {tok} ({s:.2f})")
    return "\n".join(lines[:40])

def decide(expected_words: List[str], tokens: List[str], strikes: Dict[str, Dict[str, int]], key: str) -> CheckResult:
    aligned, avg = align_expected_to_tokens(expected_words, tokens)
    missing = [w for (w, tok, s) in aligned if tok is None]
    weak = [(w, tok, s) for (w, tok, s) in aligned if tok is not None and s < OK_SIM_THRESHOLD]

    # For long ayah: never STOP (too many false alarms)
    if len(expected_words) > STOP_MAX_WORDS:
        if avg >= OK_SIM_THRESHOLD and not missing:
            strikes[key] = {}
            return CheckResult("OK", "✅ Looks OK (quiet).", avg, None, "")
        return CheckResult("LOW", "🟡 Low confidence for long ayah (ASR noisy). Try again.", avg, None, "")

    # If ASR is too messy, keep quiet and don't blame
    if avg < LOWCONF_THRESHOLD:
        return CheckResult("LOW", "🟡 Low confidence (ASR noisy). Try again.", avg, None, "")

    if not missing and len(weak) <= 1:
        strikes[key] = {}
        return CheckResult("OK", "✅ Looks OK (quiet).", avg, None, "")

    first_issue = missing[0] if missing else (weak[0][0] if weak else None)

    # Strike tracking
    if first_issue:
        strikes.setdefault(key, {})
        strikes[key][first_issue] = strikes[key].get(first_issue, 0) + 1

        # STOP only if repeated a lot AND avg is high
        if strikes[key][first_issue] >= STRIKES_TO_STOP and avg >= STOP_MIN_AVG_SIM:
            return CheckResult(
                "STOP",
                f"🛑 Possible mistake near: **{first_issue}** (repeated {STRIKES_TO_STOP}x)",
                avg,
                first_issue,
                format_alignment(aligned)
            )

    return CheckResult("LOW", "🟡 Not confident enough to stop. Try once more.", avg, first_issue, "")

# -----------------------------
# Logging
# -----------------------------
def init_state():
    return {"strikes": {}, "log": []}

def build_log_text(log_rows: List[dict]) -> str:
    if not log_rows:
        return "No logs yet."
    lines = []
    lines.append("QURAN REVISION LOG (MVP)")
    lines.append("-" * 40)
    for i, r in enumerate(log_rows, 1):
        lines.append(f"{i}) time={r['ts']} | ayah={r['ayah']} | app={r['app_status']} | note={r['app_note']}")
        lines.append(f"   asr={r['asr']}")
        lines.append(f"   user={r['user_feedback']} | word={r['user_word']} | comment={r['user_comment']}")
        lines.append("-" * 40)
    return "\n".join(lines)

# -----------------------------
# Gradio functions
# -----------------------------
def run_check(ayah_key: str, audio: Tuple[int, np.ndarray], state: dict):
    if state is None:
        state = init_state()

    if ayah_key not in QURAN:
        return "Choose a valid start point.", "", state

    sr, y = audio
    if y is None:
        return "Please record audio.", "", state

    expected = QURAN[ayah_key]
    exp_words = arabic_words(expected)

    asr_raw = run_asr(y, sr)
    toks = clean_tokens(asr_raw)

    result = decide(exp_words, toks, state["strikes"], ayah_key)

    out_lines = []
    out_lines.append(f"Start: {ayah_key}")
    out_lines.append(f"Expected: {expected}")
    out_lines.append("")
    out_lines.append(f"ASR: {asr_raw}")
    out_lines.append("")
    out_lines.append(result.note)
    if result.status == "STOP" and result.details:
        out_lines.append("")
        out_lines.append(result.details)

    # Store last check (for log finalize step)
    state["last"] = {
        "ayah": ayah_key,
        "expected": expected,
        "asr": asr_raw,
        "app_status": result.status,
        "app_note": result.note,
    }

    # Return: result text + show "awaiting feedback" hint
    hint = "Now choose your feedback below and click: Save This Attempt."
    return "\n".join(out_lines), hint, state

def save_attempt(user_feedback: str, user_word: str, user_comment: str, state: dict):
    if state is None:
        state = init_state()

    last = state.get("last")
    if not last:
        return "No attempt to save yet. First click: Check recitation.", build_log_text(state["log"]), state

    row = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ayah": last["ayah"],
        "asr": last["asr"],
        "app_status": last["app_status"],
        "app_note": last["app_note"],
        "user_feedback": user_feedback or "N/A",
        "user_word": (user_word or "").strip(),
        "user_comment": (user_comment or "").strip(),
    }
    state["log"].append(row)

    # reset last so each check must be saved explicitly
    state["last"] = None

    msg = "✅ Saved to log. Do next attempt."
    return msg, build_log_text(state["log"]), state

def clear_log(state: dict):
    state = init_state()
    return "Log cleared.", build_log_text(state["log"]), state

# -----------------------------
# UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## Qur’an Revision MVP (Auto Logger + Quiet Mode)")
    gr.Markdown("This version stays quiet. It only stops on repeated + confident issues, and it auto-builds logs.")

    state = gr.State(init_state())

    with gr.Row():
        ayah = gr.Dropdown(choices=list(QURAN.keys()), value="1:1", label="Start point")
    audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Record / Upload")

    btn_check = gr.Button("Check recitation")
    result_box = gr.Textbox(label="Result", lines=14)
    hint_box = gr.Textbox(label="Next", lines=1)

    gr.Markdown("### User Feedback (required for logging)")
    with gr.Row():
        feedback = gr.Dropdown(
            choices=[
                "✅ I recited correctly",
                "🧪 I intentionally skipped a word",
                "🧪 I intentionally said a wrong word",
                "🤷 Not sure / ASR issue",
            ],
            value="✅ I recited correctly",
            label="What did YOU do?"
        )
    with gr.Row():
        word_box = gr.Textbox(label="Which word? (optional)", placeholder="e.g., الرحمن / رب / العالمين")
    comment_box = gr.Textbox(label="Extra note (optional)", placeholder="e.g., I paused / I repeated / mic noise", lines=2)

    btn_save = gr.Button("Save This Attempt to Log")
    save_msg = gr.Textbox(label="Save status", lines=1)

    gr.Markdown("### Session Log")
    log_box = gr.Textbox(label="Log (copy & paste here)", lines=14)

    with gr.Row():
        btn_copy = gr.Button("Copy Log to Clipboard")
        btn_clear = gr.Button("Clear Log")

    # actions
    btn_check.click(run_check, inputs=[ayah, audio, state], outputs=[result_box, hint_box, state])
    btn_save.click(save_attempt, inputs=[feedback, word_box, comment_box, state], outputs=[save_msg, log_box, state])
    btn_clear.click(clear_log, inputs=[state], outputs=[save_msg, log_box, state])

    # Copy via JS
    btn_copy.click(
        fn=None,
        inputs=[log_box],
        outputs=[],
        _js="(text)=>{navigator.clipboard.writeText(text || '');}"
    )

demo.queue().launch()