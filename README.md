# NeoMTL-NLP — Chinese Neonatal Diagnosis Entity Extraction Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Open-source release of the **LLM-augmented entity-extraction pipeline** and **neonatal disease dictionary** used in:

> **Admission Diagnosis Text Dominates Early Prediction of Severe Neonatal Outcomes — A Multi-Task Deep Learning Study of 11,377 NICU Admissions** *(Scientific Reports, under review, 2026)*

---

## What this repository contains

```
NeoMTL-NLP/
├── src/
│   └── p1_nlp_extractor.py          ← 4-layer extraction pipeline (rule + LLM)
├── data/
│   └── disease_dict_v1.json         ← curated neonatal disease dictionary
│                                      (200+ Chinese surface forms,
│                                       ICD-10 P-chapter mapping,
│                                       attitude / modifier conflict rules)
├── examples/
│   └── example_discharge_diagnoses.jsonl   ← 8 synthetic demo records
├── requirements.txt
├── LICENSE  (MIT)
└── README.md  (this file)
```

> **Note on patient data.** The raw 11,377-record cohort is **not** released here — it contains identifiable hospital records and is governed by Handan Central Hospital IRB No. HDYY-LW-25053. The repository ships with eight fully synthetic demo records that exercise every code path in the pipeline. De-identified analytic data may be requested from the corresponding authors, subject to local IRB and data-protection requirements.

---

## Pipeline overview

The extractor maps a free-text Chinese neonatal discharge diagnosis (`discharge_dx_text`) to six binary disease labels:

| Label             | ICD-10 chapter | English             |
|-------------------|----------------|---------------------|
| `dx_jaundice`     | P59            | Neonatal jaundice / hyperbilirubinemia |
| `dx_chd`          | Q20–Q24        | Congenital heart disease |
| `dx_preterm_lbw`  | P07            | Preterm / low birth weight |
| `dx_sepsis`       | P36            | Neonatal sepsis     |
| `dx_rds`          | P22.0          | Respiratory distress syndrome |
| `dx_pneumonia`    | P23            | Neonatal pneumonia  |

Four-layer architecture:

1. **Layer 1 — Rule pre-processing.** Sentence splitting on Chinese commas / spaces / semicolons; full-width / half-width normalization; punctuation harmonization; traditional → simplified conversion.
2. **Layer 2 — LLM entity recognition.** Few-shot prompt sent to a Chinese medical LLM (Qwen2.5-Med, HuatuoGPT-II, or any OpenAI / Anthropic API endpoint) returning `(disease, attitude, modifier)` triples. Attitude = {`positive`, `suspected`, `negated`}; modifier = {`active`, `post_op`, `resolved`, `historical`}.
3. **Layer 3 — Dictionary normalization.** Surface forms are mapped to the six head labels via `data/disease_dict_v1.json`. Conflict-resolution rules handle e.g. *"宫内感染性肺炎"* → `pneumonia` only (not `sepsis`); patent ductus arteriosus *"已关闭"* / *"术后"* → excluded from active CHD.
4. **Layer 4 — Deterministic rule fallback.** Pure-regex extractor (`extract_with_rules`) used when LLM unavailable. Macro F1 = **0.989** on the cohort (n = 11,377) and = **0.985** on the 100-case manual gold-standard pilot.

---

## Quick start

### 1. Install

```bash
git clone https://github.com/<USER_OR_ORG>/NeoMTL-NLP.git
cd NeoMTL-NLP
pip install -r requirements.txt
```

### 2. Rule-based extraction (no LLM required)

```python
from src.p1_nlp_extractor import P1NLPExtractor
import json

extractor = P1NLPExtractor(
    dict_path="data/disease_dict_v1.json",
    backend="rules"          # pure rule-based; no LLM dependency
)

text = "新生儿呼吸窘迫综合征 新生儿呼吸衰竭 极低出生体重儿 新生儿败血症"
labels = extractor.extract(text)
print(labels)
# {'dx_jaundice': 0, 'dx_chd': 0, 'dx_preterm_lbw': 1,
#  'dx_sepsis': 1, 'dx_rds': 1, 'dx_pneumonia': 0}
```

### 3. LLM-augmented extraction

Set the backend to your preferred provider:

```python
extractor = P1NLPExtractor(
    dict_path="data/disease_dict_v1.json",
    backend="anthropic",          # or "openai", "transformers"
    model="claude-3-5-sonnet",    # or "gpt-4o-mini", or local HF model id
)
```

API keys are read from environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) or set via `extractor.client = ...`.

### 4. Batch processing the demo file

```bash
python -m src.p1_nlp_extractor \
    --input  examples/example_discharge_diagnoses.jsonl \
    --output examples/example_predictions.jsonl \
    --dict   data/disease_dict_v1.json \
    --backend rules
```

---

## Reproducibility

The full evaluation pipeline used to compute the macro F1 = 0.989 cohort number is **deterministic** (random_state = 42 throughout). The training/validation/test split, TF-IDF hyperparameters (`char_wb`, n-gram 2–4, min_df = 5, max_df = 0.95, max_features = 3000, sublinear_tf = True), and L2 logistic regression baseline (C = 1.0, class_weight = `balanced`) are documented in the manuscript Methods §2.5 and reproducible from `scripts/run_figure_s2_sweep.py` in the manuscript's supplementary package.

The pilot-validation set (n = 100, manually curated) and full-cohort reference labels were generated using the rule-based fallback (Layer 4) cross-validated against expert review. The macro F1 numbers reported in the manuscript can be reproduced by running the extractor on any private cohort using the released dictionary.

---

## Citation

If you use this code or the disease dictionary in your work, please cite:

```bibtex
@article{neomtl2026,
  title   = {Admission Diagnosis Text Dominates Early Prediction of Severe Neonatal
             Outcomes: A Multi-Task Deep Learning Study of 11,377 NICU Admissions},
  author  = {Zhang, Xiaoxue and Quan, Yanhua and Liu, Yulong and others},
  journal = {Scientific Reports},
  year    = {2026},
  doi     = {<add upon acceptance>}
}
```

---

## Authors and contacts

* **Corresponding (clinical):**  Nan Huo, MD — Handan Central Hospital, Department of Neonatology Ward I — `18231066810@163.com`
* **Corresponding (cardiology liaison):**  Mengdan Miao, MD — Handan Key Laboratory of Cardiac Precision Medicine — `m18032936267@163.com`
* **Corresponding (methods):**  Sirui Han, PhD — Division of Emerging Interdisciplinary Areas, Hong Kong University of Science and Technology — `siruihan@ust.hk`

For technical questions about the code, please open a GitHub issue.

---

## License

MIT (see [LICENSE](LICENSE)).

The neonatal disease dictionary is released under the same MIT terms and may be freely adapted for academic or clinical research, with attribution.
