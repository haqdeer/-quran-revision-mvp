# Current Features (As of MVP Stage)

## Implemented
- Ayah-by-ayah revision (manual start point selection)
- Arabic speech recognition using wav2vec2 (Arabic model)
- Word-level comparison against Quran text
- Detection of:
  - Missing words
  - Wrong / uncertain words
  - Skipped words
- Repeat tolerance (handles repeated words when user hesitates)
- Basic “stuck” detection using trailing silence
- Clean result output with copy-to-clipboard support

## Supported Scope
- Surah Al-Fatihah
- Ayah 1 to 3
- One ayah per recitation (intentionally enforced)

## Not Implemented Yet
- Tajweed rule checking
- Madd length checking
- Harakat-level (zabar/zair/pesh) strict validation
- Continuous multi-ayah listening
- Model fine-tuning or training
