"""
predict.py – Load saved models & rules, accept new clinical notes,
             and print side‑by‑side predictions.
"""

import json, re, os, spacy

# 1. Load rules
RULES = {}
for char in ["language_barrier", "living_alone", "cognitive_frailty", "non_adherence"]:
    path = f"rules/{char}.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            RULES[char] = json.load(f)
    else:
        print(f"Missing rules file: {path}")
        RULES[char] = None

# 2. Load NER models
CHAR_TO_LABEL = {
    "language_barrier":  "LANGUAGE_BARRIER",
    "living_alone":      "LIVING_ALONE",
    "cognitive_frailty": "COGNITIVE_FRAILTY",
    "non_adherence":     "NON_ADHERENCE",
}

models = {}
for char, label in CHAR_TO_LABEL.items():
    path = f"models_{char}"
    if os.path.exists(path):
        try:
            models[char] = spacy.load(path)
            print(f"Loaded NER model for {char}")
        except Exception as e:
            print(f"Failed to load {char}: {e}")
            models[char] = None
    else:
        print(f"Model not found: {path}")
        models[char] = None

# 3. Prediction functions
def rule_predict(text, char):
    if RULES.get(char) is None:
        return None
    t = text.lower()
    for excl in RULES[char]["exclusions"]:
        if re.search(excl, t):
            return False
    for incl in RULES[char]["inclusions"]:
        if re.search(incl, t):
            return True
    return False

def ner_predict(model, text, label):
    if model is None:
        return None
    doc = model(text)
    for ent in doc.ents:
        if ent.label_ == label:
            return True
    return False

# 4. Interactive loop
print("\nEnter a clinical note (or 'quit' to exit):")
while True:
    try:
        note = input("\n> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye.")
        break
    if note.lower() in ("quit", "exit", "q"):
        print("Goodbye.")
        break
    if not note:
        continue

    print("\n--- Side‑by‑side predictions ---")
    for char in CHAR_TO_LABEL:
        rb = rule_predict(note, char)
        nr = ner_predict(models.get(char), note, CHAR_TO_LABEL[char])
        print(f"{char:20s}  Rule‑based: {str(rb):5s}   NER: {str(nr):5s}")
    print("-" * 40)
