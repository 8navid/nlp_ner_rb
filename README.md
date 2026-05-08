# Text‑Mining Clinical Notes: Rule‑Based vs. NER for Patient Characteristics

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.2-green.svg)](https://spacy.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible implementation of the methodology described in  
**ten Hoope et al. (2025)** — *“Applying text‑mining to clinical notes:  
the identification of patient characteristics from electronic health records (EHRs)”*  
*BMC Medical Informatics and Decision Making*, 25(Suppl 3), 302.  
[https://doi.org/10.1186/s12911-025-03137-x](https://doi.org/10.1186/s12911-025-03137-x)

---

## Overview

This repository applies two text‑mining techniques to extract four patient characteristics from clinical notes:

| Characteristic | Method |
|---------------|--------|
| Language barrier | Rule‑based (SQL‑like regex queries) |
| Living alone | Transition‑based neural NER (spaCy v3.2 **default NER**, blank model) |
| Cognitive frailty | Transition‑based neural NER (spaCy v3.2 **default NER**, blank model) |
| Non‑adherence | Transition‑based neural NER (spaCy v3.2 **default NER**, blank model) |

**Key design decisions (exactly as in the paper):**
- No pre‑trained word embeddings — the NER trains from scratch on manual annotations (`spacy.blank("en")`).
- spaCy’s default NER architecture: `MultiHashEmbed → MaxoutWindowEncoder (CNN, depth=4) → TransitionBasedParser (stack/buffer + BILU actions)`.
- Training: batch compounding (1.0 → 8.0), dropout 0.2, max 30 epochs with early stopping, Adam.v1 optimizer.
- Evaluation on a **held‑out test set** and a separate **validation set**, using recall, specificity, precision, NPV, and F1-score.

---

## Repository Structure
.
├── dataset_with_entities.json # 100 synthetic clinical notes with manual entity spans
├── article.py # Main script: training + evaluation
├── predict.py # Interactive prediction on new notes
├── models_language_barrier/ # Saved spaCy NER model (after training)
├── models_living_alone/ # Saved spaCy NER model
├── models_cognitive_frailty/ # Saved spaCy NER model
├── models_non_adherence/ # Saved spaCy NER model
├── rules/ # JSON files with inclusion/exclusion regex patterns
│ ├── language_barrier.json
│ ├── living_alone.json
│ ├── cognitive_frailty.json
│ └── non_adherence.json
├── report.txt # Training loss + evaluation metrics
└── README.md

---
## Data Format

Each entry in `dataset_with_entities.json` follows the format:

```json
{
  "patient_id": 1,
  "note": "Patient speaks only Mandarin ...",
  "labels": {
    "language_barrier": true,
    "living_alone": true,
    "cognitive_frailty": false,
    "non_adherence": true
  },
  "entities": [
    [0, 30, "LANGUAGE_BARRIER"],
    [78, 102, "LIVING_ALONE"],
    [140, 165, "NON_ADHERENCE"]
  ]
}

The entities field contains character‑level span annotations (start, end, label) — exactly as two human annotators would produce. These spans are used to train the NER models from scratch.
