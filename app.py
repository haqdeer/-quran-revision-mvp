import gradio as gr
import numpy as np
import torch
import torchaudio
import re
from difflib import SequenceMatcher
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ---------------------------
# Quran demo scope (small)
# ---------------------------
QURAN_DEMO = {
    (1, 1): "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    (1, 2): "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
    (1, 3): "الرَّحْمَٰنِ الرَّحِيمِ",
}

# ---------------------------
# Arabic helpers
# ---------------------------
AR_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")

def strip_diacritics(text: str) -> str:
    return AR_DIACRITICS_RE.sub("", text)

def normalize_arabic(text: str) -> str:
    text = strip_diacritics(text).replace("ـ", "")
    return " ".join(text.split())

def tokenize_arabic(text: str):
    return normalize_arabic(text).split()

def normalize_latin(text: str) -> str:
    text = text.lower().replace("~", " ")
    # keep only a-z and spaces
    text = re.sub(r"[^a-z\s]", " ", text)
    return " ".join(text.split())

def tokenize_latin(text: str):
    toks = normalize_latin(text).split()
    # IMPORTANT: drop very short junk tokens (like "alr")
    toks = [t for t in toks if len(t) >= 4]
    return toks

AR2LAT = str.maketrans({
    "ا":"a","أ":"a","إ":"i","آ":"a","ب":"b","ت":"t","ث":"t",
    "ج":"j","ح":"h","خ":"h","د":"d","ذ":"dh","ر":"r","ز":"z",
    "س":"s","ش":"sh","ص":"s","ض":"d","ط":"t","ظ":"z","ع":"a",
    "غ":"gh","ف":"f","ق":"q","ك":"k","ل":"l","م":"m","ن":"n",
    "ه":"h","و":"w","ي":"y","ة":"h","ى":"a","ء":""
})

def ar_to_proxy(word: str) -> str:
    w = strip_diacritics(word)
    w = re.sub(r"[^\u0621-\u064A]", "", w)
    return w.translate(AR2LAT)

def sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

# ---------------------------
# ASR token cleanup + repeat tolerance
# ---------------------------
def compress_repeats(tokens):
    """Remove repeated/near-repeated tokens like rab/rabi/rab..."""
    out = []
    for t in tokens:
        if not out:
            out.append(t)
            continue
        prev = out[-1]
        # if very similar, treat as repeat
        if sim(prev, t) >= 0.86:
            continue
        out.append(t)
    return out

# ---------------------------
# Audio helpers
# ---------------------------
TARGET_SR = 16000
SILENCE_RMS = 0.012  # overall RMS threshold
TRAILING_SILENCE_SEC = 1.0  # stuck threshold

def to_float_audio(y: np.ndarray) -> np.ndarray:
    if y is None:
        return y
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    else:
        y = y.astype(np.float32)
    return np.clip(y, -1.0, 1.0)

def resample(y: np.ndarray, sr: int) -> np.ndarray:
    y = to_float_audio(y)
    if int(sr) == TARGET_SR:
        return y
    yt = torch.from_numpy(y).float()
    yr = torchaudio.functional.resample(yt, int(sr), TARGET_SR)
    return yr.numpy()

def detect_overall_silence(y: np.ndarray) -> bool:
    y = to_float_audio(y)
    if y is None or len(y) == 0:
        return True
    rms = float(np.sqrt(np.mean(y**2)))
    return rms < SILENCE_RMS

def trailing_silence_seconds(y: np.ndarray) -> float:
    """Detect how much silence exists at end of clip (very simple)."""
    y = to_float_audio(y)
    if y is None or len(y) == 0:
        return 0.0
    # compute short-window energy
    win = int(0.02 * TARGET_SR)  # 20ms
    if win <= 0:
        return 0.0
    step = win
    energies = []
    for i in range(0, len(y) - win + 1, step):
        chunk = y[i:i+win]
        energies.append(float(np.sqrt(np.mean(chunk**2))))
    if not energies:
        return 0.0

    # last index where energy above threshold
    last_voice = -1
    for idx, e in enumerate(energies):
        if e >= SILENCE_RMS:
            last_voice = idx

    if last_voice == -1:
        return len(y) / TARGET_SR

    tail_chunks = (len(energies) - 1) - last_voice
    return tail_chunks * (step / TARGET_SR)

# ---------------------------
# ASR model
# ---------------------------
MODEL_ID = "elgeish/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()

# ---------------------------
# DP alignment (same idea, stricter)
# ---------------------------
def align_dp(expected_proxy, asr_tokens, base_thr=0.58):
    n = len(expected_proxy)
    m = len(asr_tokens)

    INS_COST = 0.95
    DEL_COST = 0.95

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i-1][0] + DEL_COST
        back[i][0] = ("DEL", i-1, 0)

    for j in range(1, m + 1):
        dp[0][j] = dp[0][j-1] + INS_COST
        back[0][j] = ("INS", 0, j-1)

    def sub_cost(e, a):
        # IMPORTANT: prevent short ASR token matching long expected word
        if len(a) < 4:
            return 1.35

        # dynamic threshold by expected length
        s = sim(e, a)
        thr = base_thr
        if len(e) >= 7:
            thr = base_thr - 0.02  # allow a bit flexibility for longer words
        if len(e) <= 3:
            thr = base_thr + 0.05  # short expected word must be stricter

        if s >= thr:
            return 1.0 - s
        return 1.35

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            e = expected_proxy[i-1]
            a = asr_tokens[j-1]
            c_sub = dp[i-1][j-1] + sub_cost(e, a)
            c_del = dp[i-1][j] + DEL_COST
            c_ins = dp[i][j-1] + INS_COST

            best = min(c_sub, c_del, c_ins)
            dp[i][j] = best
            if best == c_sub:
                back[i][j] = ("SUB", i-1, j-1)
            elif best == c_del:
                back[i][j] = ("DEL", i-1, j)
            else:
                back[i][j] = ("INS", i, j-1)

    matched = []
    missing_exp = []
    extra_asr = []

    i, j = n, m
    while i > 0 or j > 0:
        step = back[i][j]
        if step is None:
            break
        kind, pi, pj = step
        if kind == "SUB":
            exp_i, asr_j = pi, pj
            s = sim(expected_proxy[exp_i], asr_tokens[asr_j])
            matched.append((exp_i, asr_j, s))
            i, j = exp_i, asr_j
        elif kind == "DEL":
            exp_i = pi
            missing_exp.append(exp_i)
            i = exp_i
        elif kind == "INS":
            asr_j = pj
            extra_asr.append(asr_j)
            j = asr_j

    matched.reverse()
    missing_exp.reverse()
    extra_asr.reverse()
    return matched, missing_exp, extra_asr

def build_word_report(expected_words, expected_proxy, asr_tokens, matched, conf_thr=0.62):
    best_for_exp = {i: None for i in range(len(expected_words))}
    for exp_i, asr_j, s in matched:
        cur = best_for_exp.get(exp_i)
        if cur is None or s > cur[1]:
            best_for_exp[exp_i] = (asr_j, s)

    missing = []
    wrong = []
    ok = []
    detail_lines = []

    for i, w in enumerate(expected_words):
        item = best_for_exp.get(i)
        if item is None:
            missing.append(w)
            detail_lines.append(f"- {w}  ->  ❌ (no token aligned)")
            continue

        asr_j, s = item
        tok = asr_tokens[asr_j] if 0 <= asr_j < len(asr_tokens) else "?"

        # extra rule: long expected word can't be satisfied by too-short token
        if len(expected_proxy[i]) >= 6 and len(tok) < 5:
            missing.append(w)
            detail_lines.append(f"- {w}  ->  ❌ (token too short: {tok})")
            continue

        if s >= conf_thr:
            ok.append(w)
            detail_lines.append(f"- {w}  ->  ✅ {tok}  (sim {s:.2f})")
        else:
            wrong.append((w, tok, s))
            detail_lines.append(f"- {w}  ->  ⚠️ {tok}  (sim {s:.2f})")

    return ok, missing, wrong, detail_lines

# ---------------------------
# Main function
# ---------------------------
def check_recitation(surah, ayah, audio):
    surah = int(surah)
    ayah = int(ayah)
    key = (surah, ayah)
    expected = QURAN_DEMO.get(key)
    if expected is None:
        return "❌ Demo scope: only Surah 1 Ayah 1–3 available right now."

    expected_words = tokenize_arabic(expected)
    expected_proxy = [ar_to_proxy(w) for w in expected_words]

    if audio is None:
        return (
            f"✅ Start point set: {surah}:{ayah}\n"
            f"Expected: {expected}\n\n"
            f"Rule: Select ONE ayah and recite ONLY that ayah."
        )

    sr, y = audio
    y = resample(y, int(sr))
    duration = len(y) / TARGET_SR

    overall_silent = detect_overall_silence(y)
    tail_sil = trailing_silence_seconds(y)

    inputs = processor(y, sampling_rate=TARGET_SR, return_tensors="pt")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    pred_ids = torch.argmax(logits, dim=-1)
    raw = processor.batch_decode(pred_ids)[0]

    asr_tokens = tokenize_latin(raw)
    asr_tokens = compress_repeats(asr_tokens)

    matched, _, _ = align_dp(expected_proxy, asr_tokens, base_thr=0.58)

    ok, missing, wrong, detail_lines = build_word_report(
        expected_words, expected_proxy, asr_tokens, matched, conf_thr=0.62
    )

    # pointer = first expected word not confidently OK
    pointer = 0
    while pointer < len(expected_words) and expected_words[pointer] in ok:
        pointer += 1

    lines = []
    lines.append(f"✅ Start point: {surah}:{ayah}")
    lines.append(f"Expected (with harakaat): {expected}")
    lines.append(f"Expected words: {expected_words}")
    lines.append(f"Expected proxy: {expected_proxy}")
    lines.append("")
    lines.append(f"🎙️ Duration: {duration:.2f}s | SR: {TARGET_SR}")
    lines.append(f"📝 ASR Raw: {raw}")
    lines.append(f"🧩 ASR Tokens (cleaned): {asr_tokens}")
    lines.append("")
    lines.append("---")
    lines.append(f"Matched (OK): {len(ok)}/{len(expected_words)} | ASR tokens: {len(asr_tokens)}")

    if missing:
        lines.append("❌ Missing word(s): " + " , ".join(missing))
    if wrong:
        lines.append("⚠️ Wrong/uncertain word(s): " + " , ".join([f"{w}⇢{t}({s:.2f})" for (w,t,s) in wrong]))

    if not missing and not wrong:
        lines.append("✅ Looks OK (MVP).")

    if pointer < len(expected_words):
        lines.append(f"➡️ Next expected word (hint): **{expected_words[pointer]}**")
    else:
        lines.append("➡️ Completed expected words (hint).")

    lines.append("")
    lines.append("🔍 Per-word alignment:")
    lines.extend(detail_lines)

    # basic stuck detection (end pause)
    if overall_silent:
        lines.append("")
        lines.append("⏸️ Mostly silence detected. Try reciting again clearly.")
    elif tail_sil >= TRAILING_SILENCE_SEC and pointer < len(expected_words):
        lines.append("")
        lines.append(f"🧠 Possible stuck detected (you paused ~{tail_sil:.1f}s at the end).")
        lines.append(f"Hint: next word is **{expected_words[pointer]}**")

    lines.append("")
    lines.append("Rule: Select ONE ayah and recite ONLY that ayah.")
    lines.append("Next step after this: we’ll add a tiny ‘training log’ (save your attempts) without heavy model training.")

    return "\n".join(lines)

# ---------------------------
# UI + Copy (working)
# ---------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📖 Quran Revision App — Step 10.3 (No tiny-token cheating + repeat + stuck)")

    with gr.Row():
        surah = gr.Number(value=1, label="Surah")
        ayah = gr.Number(value=1, label="Ayah")

    audio = gr.Audio(
        sources=["microphone"],
        type="numpy",
        label="Recite (single ayah)"
    )

    output = gr.Textbox(lines=26, label="Result", elem_id="result_box")

    btn = gr.Button("🎙️ Check Recitation")
    btn.click(check_recitation, [surah, ayah, audio], output)

    gr.HTML("""
    <div style="margin-top:10px;">
      <button style="padding:10px 14px; border-radius:10px; cursor:pointer;"
        onclick="
          const ta = document.querySelector('#result_box textarea');
          const txt = ta ? ta.value : '';
          navigator.clipboard.writeText(txt).then(
            () => alert('✅ Copied to clipboard'),
            () => alert('❌ Copy failed (permission)')
          );
        ">
        📋 Copy Result
      </button>
    </div>
    """)

demo.launch()
