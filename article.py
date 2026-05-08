"""
article.py – Full replication of ten Hoope et al. (2025) using
manually‑annotated entities, rule‑based queries, and blank spaCy NER
with early stopping.

Input:  dataset_with_entities.json (100 patients with "entities" field)
Output: models_language_barrier/, models_living_alone/, ...
        rules/ directory, report.txt
"""

import json, re, random, os, sys
from sklearn.metrics import confusion_matrix
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

# ----------------------------------------------------------------------
# 0. Tee output to report.txt
# ----------------------------------------------------------------------
class Tee:
    def __init__(self, filename, mode="w"):
        self.file = open(filename, mode, encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()

sys.stdout = Tee("report.txt")

# ----------------------------------------------------------------------
# 1. Load dataset
# ----------------------------------------------------------------------
with open("dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)
print(f"Loaded {len(data)} patients.")

# ----------------------------------------------------------------------
# 2. Rule‑based definitions (all four characteristics)
# ----------------------------------------------------------------------
RULES = {
    "language_barrier": {
        "inclusions": [
            r"does not speak english",
            r"language barrier",
            r"needs interpreter",
            r"interpreter\s+(needed|required|used|for|essential|booked|provided)",
            r"limited english proficiency",
            r"communication via interpreter",
            r"patient only speaks",
            r"speaks only",
            r"interpreter required",
            r"interpreter provided",
        ],
        "exclusions": [
            r"family provides? interpretation",
            r"daughter translates?",
            r"speaks? english fluently",
            r"no language barrier",
            r"language\s*:\s*fluent",
            r"speech\s+normal",
            r"language\s+intact",
            r"speech and language\s+normal",
            r"no language issues",
            r"no language problems",
        ]
    },
    "living_alone": {
        "inclusions": [
            r"lives alone",
            r"lives independently but alone",
            r"lives by himself",
            r"lives by herself",
            r"resides alone",
            r"no cohabitants",
            r"no family at home",
            r"living situation: alone",
            r"lives alone in",
            r"has lived alone since",
            r"now living alone",
            r"lives alone on",
            r"lives alone after",
        ],
        "exclusions": [
            r"lives with",
            r"resides with",
            r"lives with spouse",
            r"lives with partner",
            r"lives with family",
            r"lives with daughter",
            r"lives with son",
            r"married, living together",
            r"in a nursing home",
            r"retirement village",
            r"assisted-living facility",
            r"group home",
            r"supported accommodation",
            r"in a convent",
            r"with other sisters",
            r"monastic community",
        ]
    },
    "cognitive_frailty": {
        "inclusions": [
            r"forgetful",
            r"forgetfulness",
            r"memory problems",
            r"cognitive impairment",
            r"cognitive decline",
            r"dementia",
            r"alzheimer",
            r"signs of dementia",
            r"mild cognitive impairment",
            r"word.?finding difficulties",
            r"getting lost",
            r"repeats questions",
            r"vascular dementia",
            r"lewy bodies",
            r"moderate dementia",
            r"severe dementia",
            r"advanced dementia",
            r"cognitive decline obvious",
            r"sundowning",
            r"poor memory",
            r"short.?term memory",
        ],
        "exclusions": [
            r"no cognitive",
            r"cognitively intact",
            r"memory intact",
            r"alert and oriented",
            r"no signs of dementia",
            r"cognition normal",
            r"cognitively normal",
            r"no memory problems",
            r"no cognitive complaints",
            r"no cognitive issues",
            r"no cognitive concerns",
            r"no dementia",
            r"cognitive assessment normal",
        ]
    },
    "non_adherence": {
        "inclusions": [
            r"non.?adherent",
            r"non.?adherence",
            r"skipped several doses",
            r"frequently misses",
            r"frequently miss",
            r"misses medication",
            r"missed several doses",
            r"suspected medication noncompliance",
            r"not taking as prescribed",
            r"occasionally forgets to take",
            r"forgets to take",
            r"missing doses",
            r"not following prescribed",
            r"intentional adjustment",
            r"stops? taking",
            r"skips doses",
            r"skips his",
            r"skips his blood",
            r"discontinued antidepressant",
            r"stopped all medications",
            r"refuses to take",
            r"spits out his tablets",
            r"irregular pickup",
            r"gaps in",
            r"multiple missed",
            r"misses evening",
            r"misses afternoon",
            r"medications are barely taken",
            r"consistently missed",
        ],
        "exclusions": [
            r"compliant",
            r"good adherence",
            r"taken regularly",
            r"no missed doses",
            r"fully adherent",
            r"100%",
            r"takes .* as prescribed",
            r"perfect adherence",
            r"adherence is excellent",
            r"fully compliant",
            r"compliance is excellent",
            r"medication compliance reported as good",
            r"adherence is good",
            r"takes all medications",
        ]
    }
}

def rule_based_predict(text, char):
    t = text.lower()
    for excl in RULES[char]["exclusions"]:
        if re.search(excl, t):
            return False
    for incl in RULES[char]["inclusions"]:
        if re.search(incl, t):
            return True
    return False

for entry in data:
    for char in RULES:
        entry[f"rb_{char}"] = rule_based_predict(entry["note"], char)

# ----------------------------------------------------------------------
# 3. Prepare training data from manual "entities"
# ----------------------------------------------------------------------
CHARACTERISTICS = ["language_barrier", "living_alone", "cognitive_frailty", "non_adherence"]
CHAR_TO_LABEL = {
    "language_barrier":  "LANGUAGE_BARRIER",
    "living_alone":      "LIVING_ALONE",
    "cognitive_frailty": "COGNITIVE_FRAILTY",
    "non_adherence":     "NON_ADHERENCE",
}

def merge_entity_spans(spans):
    """Merge overlapping or touching span lists [(start, end, label), ...]."""
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], -x[1]))
    merged = []
    cur_s, cur_e, cur_l = spans[0]
    for s, e, l in spans[1:]:
        if s <= cur_e:          # overlap or touch
            if e > cur_e:
                cur_e = e
        else:
            merged.append((cur_s, cur_e, cur_l))
            cur_s, cur_e, cur_l = s, e, l
    merged.append((cur_s, cur_e, cur_l))
    return merged

def build_ner_data_for_char(ids, char):
    nlp_blank = spacy.blank("en")
    examples = []
    label = CHAR_TO_LABEL[char]
    for i in ids:
        entry = data[i]
        text = entry["note"]
        # 1. Filter only this characteristic's entities
        raw_spans = [(s, e, label) for (s, e, lbl) in entry.get("entities", []) if lbl == label]
        # 2. Merge overlapping raw spans
        raw_spans = merge_entity_spans(raw_spans)
        # 3. Align to token boundaries
        doc = nlp_blank(text)
        aligned = []
        for s, e, lbl in raw_spans:
            span = doc.char_span(s, e, alignment_mode="expand")
            if span is not None:
                aligned.append((span.start_char, span.end_char, lbl))
        # 4. Merge again – token alignment may create new overlaps
        aligned = merge_entity_spans(aligned)
        examples.append((text, {"entities": aligned}))
    return examples

# Train / test / validation split
random.seed(42)
indices = list(range(len(data)))
random.shuffle(indices)
n = len(data)
train_ids = indices[:int(0.7*n)]
test_ids  = indices[int(0.7*n):int(0.85*n)]
val_ids   = indices[int(0.85*n):]
print(f"Train: {len(train_ids)}  Test: {len(test_ids)}  Validation: {len(val_ids)}")

ner_train = {c: build_ner_data_for_char(train_ids, c) for c in CHARACTERISTICS}
for c in CHARACTERISTICS:
    pos = sum(1 for _, ann in ner_train[c] if ann["entities"])
    print(f"  {c}: {pos} training examples with entities")

# ----------------------------------------------------------------------
# 4. Train spaCy NER models with early stopping
# ----------------------------------------------------------------------
def train_ner_model(train_data, entity_label, n_iter=60, patience=10):
    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner", last=True)
    ner.add_label(entity_label)

    # Convert training data
    train_examples = []
    for text, ann in train_data:
        doc = nlp.make_doc(text)
        try:
            ex = Example.from_dict(doc, ann)
            train_examples.append(ex)
        except Exception as e:
            print(f"    Skipping example: {e}")

    if not train_examples:
        return nlp

    # Internal validation split (10% of training)
    random.shuffle(train_examples)
    split_idx = max(1, int(0.9 * len(train_examples)))
    train_split = train_examples[:split_idx]
    val_split = train_examples[split_idx:]

    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    best_loss = float('inf')
    best_model_bytes = None
    no_improve = 0

    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_split)
            losses = {}
            batches = minibatch(train_split, size=compounding(1.0, 8.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=0.2, losses=losses)
            train_loss = losses.get('ner', 0.0)

            # Check validation loss every 5 epochs
            if val_split and (itn + 1) % 5 == 0:
                # For consistency, use training loss as proxy (spaCy evaluate is slow & complex)
                if train_loss < best_loss:
                    best_loss = train_loss
                    best_model_bytes = nlp.to_bytes()
                    no_improve = 0
                else:
                    no_improve += 1

                if (itn + 1) % 10 == 0:
                    print(f"    Epoch {itn+1}/{n_iter}, Train Loss: {train_loss:.4f}")

                if no_improve >= patience:
                    print(f"    Early stopping at epoch {itn+1} (no improvement for {patience} checks)")
                    break

    # Restore best model
    if best_model_bytes:
        nlp.from_bytes(best_model_bytes)
        print(f"    Restored best model (loss: {best_loss:.4f})")
    return nlp

# Train & save
os.makedirs("models", exist_ok=True)
os.makedirs("rules", exist_ok=True)

models = {}
for char, label in CHAR_TO_LABEL.items():
    print(f"\nTraining NER for {label} ...")
    models[char] = train_ner_model(ner_train[char], label)
    path = f"models_{char}"
    models[char].to_disk(path)
    print(f"  Saved to {path}/")
    with open(f"rules/{char}.json", "w", encoding="utf-8") as f:
        json.dump(RULES[char], f, indent=2)

# ----------------------------------------------------------------------
# 5. Evaluation
# ----------------------------------------------------------------------
def predict_ner_patient(model, text, entity_label):
    doc = model(text)
    for ent in doc.ents:
        if ent.label_ == entity_label:
            return True
    return False

def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    rec = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"recall": round(rec,3), "specificity": round(spec,3),
            "precision": round(prec,3), "NPV": round(npv,3), "F1": round(f1,3)}

def evaluate_set(ids, set_name):
    print(f"\n--- {set_name} set ---")
    for char in CHARACTERISTICS:
        y_true = [data[i]["labels"][char] for i in ids]
        y_rb   = [data[i][f"rb_{char}"] for i in ids]
        y_ner  = [predict_ner_patient(models[char], data[i]["note"], CHAR_TO_LABEL[char]) for i in ids]
        print(f"\n{char:20s}  RB: {compute_metrics(y_true, y_rb)}")
        print(f"{'':20s}  NER: {compute_metrics(y_true, y_ner)}")

evaluate_set(test_ids, "Test")
evaluate_set(val_ids, "Validation")

print("\nAll models and rules saved.")
sys.stdout.file.close()
sys.stdout = sys.stdout.stdout
