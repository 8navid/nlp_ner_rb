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
