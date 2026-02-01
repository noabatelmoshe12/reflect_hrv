# reflect_hrv — Practicum Data Science Project (Neurotech)

A reproducible ECG → HRV processing pipeline developed during a Data Science practicum in a neurotech environment.
The goal is to turn raw physiological recordings into clean, analysis-ready feature tables (e.g., HRV metrics) for downstream research and modeling.

> **Confidentiality note:** Raw participant data and any sensitive/internal assets are not included in this repository.

---

## Project context
This repository represents my practicum work at **Reflect Innovation** (neurotech).
It focuses on building a clean and repeatable processing workflow for physiological signals, with an emphasis on:
- reliability (same inputs → same outputs),
- scalability (batch processing across participants),
- and clear outputs for analysis (tidy tables + logs).

---

## Goals
- Validate and organize raw recordings (file structure, missing values, basic checks)
- Preprocess ECG signals (artifact handling / filtering where relevant)
- Detect R-peaks and derive HRV metrics
- Export analysis-ready outputs (CSV tables + optional logs)

---

## Repository structure
- `src/` – core pipeline code (processing modules / scripts)
- `raw/` – local raw data folder (**not committed**; ignored by git)
- `outputs/` – generated outputs (recommended; can be partially committed if anonymized)
- `requirements.txt` – Python dependencies
- `README.md` – project overview and usage

Suggested local structure:
```text
reflect_hrv/
  raw/
  src/
  outputs/
  requirements.txt
  README.md

---

## Code overview

### `src/ecg_batch_processor.py` (entry point)
Batch orchestration script that runs the pipeline over multiple subjects/recordings.

**What it does:**
- Traverses the raw data directory (multiple recordings)
- Manages input/output paths and folder creation
- Calls the ECG→HRV logic for each recording
- Aggregates results into structured tables (e.g., CSV)
- Produces consistent outputs across the dataset (repeatable pipeline)

**Why it exists:**
A practicum dataset usually includes many sessions/participants. This script makes the workflow scalable and reduces manual work/errors.

---

### `src/ecg_hrv_processor_v2.py` (core processing logic)
Core module that implements ECG preprocessing and HRV feature extraction.

**What it does:**
- Loads an ECG signal from file (per recording)
- Applies preprocessing (e.g., filtering / artifact handling / resampling if needed)
- Detects R-peaks
- Computes HRV features (time-domain and/or frequency-domain metrics)
- Returns a clean feature representation per recording

**Why it exists:**
Separating “core logic” from “batch orchestration” keeps the code modular, testable, and easier to maintain.

---

## Requirements
- Python 3.9+ (or similar)
- Dependencies are listed in `requirements.txt` (e.g., numpy, pandas, matplotlib, neurokit2)

