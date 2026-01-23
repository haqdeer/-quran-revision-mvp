# Development Journey (What We Have Done So Far)

## Phase 1 – Basic ASR
- Arabic ASR integrated
- Audio recording via Gradio

## Phase 2 – Naive Matching (Rejected)
- Simple greedy matching
- Caused cascading errors
- Removed

## Phase 3 – DP Alignment (Accepted)
- Dynamic Programming alignment implemented
- Handles:
  - Skipped words
  - Word order issues
  - ASR merging/splitting words

## Phase 4 – Robustness Improvements
- Removed tiny-token cheating (e.g. "alr")
- Added repeat compression
- Added stuck detection using silence
- Added per-word alignment report

## Current Status
Logic is stable enough to:
- Collect meaningful training data
- Identify realistic Hafiz mistakes
