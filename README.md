# Quran Revision MVP

A minimal Quran revision app for Huffaz.

## What this project does
- Ayah-by-ayah Quran revision
- Uses Arabic speech recognition (ASR)
- Detects:
  - Missing words
  - Wrong / uncertain words (e.g. عالمون vs العالمين)
  - Repetition and getting stuck (basic)
- Designed as an MVP (logic-first, training-ready)

## Tech stack
- Python
- HuggingFace Transformers (wav2vec2 Arabic ASR)
- PyTorch + torchaudio
- Gradio (web UI)

## Current scope
- Surah Al-Fatihah (Ayah 1–3)
- Single ayah recitation only
- No tajweed checking yet
- No heavy model training yet

## Vision
This project will evolve into a full Quran revision assistant for Huffaz,
with harakat-level accuracy and Quran-specific training.

## Status
🚧 MVP under active development
