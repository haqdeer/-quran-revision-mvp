# Architecture & Logic

## Technology Stack
- Python
- HuggingFace Transformers
- wav2vec2 Arabic ASR model
- PyTorch + torchaudio
- Gradio (web interface)

## High-Level Flow
1. User selects Surah & Ayah
2. User recites from memory
3. Audio is resampled and normalized
4. ASR converts speech → text
5. ASR output is cleaned and tokenized
6. Expected Quran words are tokenized
7. Dynamic Programming (DP) alignment matches:
   - expected words
   - ASR tokens
8. Each word is classified as:
   - OK
   - Missing
   - Wrong / uncertain
9. If pause detected → “stuck” hint shown

## Design Principle
The system is designed to be:
- Conservative (does not over-judge)
- Hafiz-friendly
- Future-ready for training
