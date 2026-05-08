# Clinical Text Mining: Rule-Based vs NER for Patient Characteristics

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![spaCy](https://img.shields.io/badge/spaCy-3.2-green.svg)](https://spacy.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A reproducible implementation of **ten Hoope et al. (2025)** — *"Applying text-mining to clinical notes: the identification of patient characteristics from electronic health records (EHRs)"* — *BMC Medical Informatics and Decision Making*, 25(Suppl 3), 302. [https://doi.org/10.1186/s12911-025-03137-x](https://doi.org/10.1186/s12911-025-03137-x)

This repository compares **rule-based SQL-like queries** with **spaCy v3.2's default transition-based NER architecture** (`MultiHashEmbed` token encoder → 4-layer `MaxoutWindowEncoder` CNN with residual connections → stack/buffer parser predicting BILUO actions), trained from scratch without pre-trained embeddings on manually annotated clinical notes, to extract four patient characteristics: **language barrier**, **living alone**, **cognitive frailty**, and **non-adherence**. Evaluation uses recall, specificity, precision, NPV, and F1-score on held-out test and validation sets.

---

## Data Format

Each entry in `dataset_with_entities.json` contains a clinical note, binary labels, and character-level entity span annotations:

```json
{
  "patient_id": 1,
  "note": "Patient speaks only Mandarin. Lives alone in an apartment...",
  "labels": {
    "language_barrier": true,
    "living_alone": true,
    "cognitive_frailty": false,
    "non_adherence": true
  },
  "entities": [
    [0, 30, "LANGUAGE_BARRIER"],
    [32, 56, "LIVING_ALONE"],
    [98, 123, "NON_ADHERENCE"]
  ]
}
```
---

## Usage

### Train Models

```bash
python article.py
```

This loads the annotated dataset, applies rule-based predictions using regex patterns, trains one blank spaCy NER model per characteristic with early stopping, evaluates on held-out test and validation sets, saves trained models and rule files, and writes a full report to `report.txt`.

### Predict on New Notes

```bash
python predict.py
```

Paste any clinical note at the prompt. The script prints side-by-side predictions from both the rule-based system and the four NER models:

```text
> Patient only speaks Arabic. Lives alone. Often forgets medications.

language_barrier      Rule‑based: True    NER: True
living_alone          Rule‑based: True    NER: True
cognitive_frailty     Rule‑based: True    NER: True
non_adherence         Rule‑based: True    NER: True
```


provided by navid danaee 8navid@gmail.com.
