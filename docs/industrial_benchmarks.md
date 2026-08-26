# Industrial Benchmark Datasets — Research Notes

This document tracks candidate industrial/process-monitoring datasets for
extending the continual-learning evaluation beyond the Tennessee Eastman
Process (TEP), why each was (or wasn't) chosen, and the exact provenance of
every download link so results can be independently reproduced.

## Status summary

| Dataset | Host | Status | mammoth dataset name |
|---|---|---|---|
| Tennessee Eastman Process (TEP) | GitHub mirror of Braatz-group data | Implemented (paper) | `tennessee-eastman` |
| Steel Plates Faults | UCI ML Repository | **Implemented** | `steel-plates-faults` |
| SECOM | UCI ML Repository | **Implemented** (time-block Class-IL trick) | `secom` |
| NASA C-MAPSS Turbofan Degradation | NASA Open Data Portal | **Implemented** (RUL-binned Class-IL) | `cmapss` |
| AI4I 2020 Predictive Maintenance | UCI ML Repository | Identified, not yet implemented — see below | not yet wired in |
| CWRU Bearing Fault | Case Western Reserve University | **Not added** — official site currently redirects through an ad-tracking domain (`match.adsrvr.org`); no verified download script added for safety reasons | n/a |

## 1. Steel Plates Faults (implemented)

- **Citation:** Buscema, M., Terzi, S., & Tastle, W. (2010). *Steel Plates
  Faults* [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C5J88N
- **Link:** https://archive.ics.uci.edu/dataset/198/steel+plates+faults
- **License:** CC BY 4.0
- **Format:** 1941 rows x 34 columns (27 numeric features + 7-column one-hot
  fault-type indicator), whitespace-separated, no header.
- **Classes:** Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps,
  Other_Faults (7 total).
- **Design decision:** to get an evenly-divisible Class-IL split, we merge the
  two rarest classes — *Stains* (72 samples) and *Dirtiness* (55 samples),
  both surface-blemish defects — into a single *Surface_Blemish* class,
  giving 6 classes total and a clean 3-task, 2-classes-per-task split:
  - Task 0: {Pastry, Z_Scratch}
  - Task 1: {K_Scatch, Surface_Blemish}
  - Task 2: {Bumps, Other_Faults}
- **Normalization:** features are z-scored using statistics computed from a
  stratified training split only (not test-set statistics), avoiding the
  per-file normalization pitfall discussed for TEP.
- **Caveat:** unlike TEP, feature vectors are static (one row per defect
  region — no genuine temporal structure). The CfC backbone treats each row
  as a length-1 sequence, so this benchmark tests whether CfC's *sparse
  wiring* still helps in the absence of sequential dynamics for it to
  exploit, complementing the temporal TEP benchmark rather than duplicating it.
- **Download:** `scripts/download_steel_plates_faults.sh`
- **Dataset loader:** `mammoth/datasets/steel_plates_faults.py`
- **Backbones:** `mammoth/backbone/SteelPlatesCfc.py`
  (`steelplatesmlp`, `steelplatescfc`)

## 2. SECOM (implemented)

- **Citation:** McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI
  Machine Learning Repository. https://doi.org/10.24432/C54305
- **Link:** https://archive.ics.uci.edu/dataset/179/secom
- **License:** CC BY 4.0
- **Format:** 1567 rows x 590 sensor features (many missing values), binary
  pass/fail label (-1 = pass, 1 = fail) plus a per-sample timestamp.
- **Design decision:** SECOM only has 2 native classes, which does not
  support a multi-task Class-IL split on its own. We sort samples
  chronologically by their timestamp and cut the record into 3 contiguous
  "eras" of equal size, then treat each era's pass/fail pair as a *separate*
  pair of classes ("task-specific class copies"): 3 tasks x 2 classes = 6
  total classes, one pass/fail pair per era. This is a deliberate
  construction to fit mammoth's Class-IL infrastructure, not a claim that
  SECOM has 6 natural classes.
- **Preprocessing:** feature columns with >50% missing values are dropped
  (562 of 590 survive); remaining NaNs are mean-imputed and all features
  z-scored using statistics computed from each era's own training split only.
- **Download:** `scripts/download_secom.sh`
- **Dataset loader:** `mammoth/datasets/secom.py`
- **Backbones:** `mammoth/backbone/SecomCfc.py` (`secommlp`, `secomcfc`)
- **Verified:** 1-epoch SGD smoke test runs and produces above-chance
  Task-IL accuracy (see repo memory / session notes for exact numbers).

## 3. NASA C-MAPSS Turbofan Engine Degradation Simulation (implemented)

- **Citation:** Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
  *Damage Propagation Modeling for Aircraft Engine Run-to-Failure
  Simulation.* In Proceedings of the 1st International Conference on
  Prognostics and Health Management (PHM08), Denver, CO.
- **Host:** NASA Open Data Portal (Prognostics Center of Excellence), **not
  UCI** — https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
- **Direct download:** https://data.nasa.gov/docs/legacy/CMAPSSData.zip
- **License:** Public domain (US Government work); no separate license text
  specified on the NASA dataset page.
- **Format:** four sub-datasets (FD001–FD004), each a space-separated text
  file with 26 columns (unit id, time in cycles, 3 operational settings, 21
  sensor readings) per operating cycle, one row per cycle per engine unit.
  FD001/FD003 run under one operating condition, FD002/FD004 under six;
  FD001/FD002 have one fault mode (HPC degradation), FD003/FD004 have two
  (HPC + fan degradation).
- **Why it fits Class-IL:** C-MAPSS is natively a **regression** benchmark —
  the task is to predict Remaining Useful Life (RUL) in cycles, not to
  classify among discrete fault types. We cap RUL at 130 cycles and bin it
  into 3 discrete health states (Healthy: RUL >= 100, Degrading: 30 <= RUL <
  100, Critical: RUL < 30), a standard RUL-capping scheme in the prognostics
  literature, and treat each of the four FD00X sub-datasets (which differ in
  operating-condition count and fault mode) as one task with its own 3
  classes: 4 tasks x 3 classes = 12 total classes.
- **Windowing:** sliding windows of 30 cycles (stride 5) over the 24 input
  channels (3 operational settings + 21 sensors) per engine unit; features
  z-scored using each FD00X subset's own training-split statistics.
- **Download:** `scripts/download_cmapss.sh` (verified working; the NASA
  "resource" landing page is HTML, the actual archive is at
  `data.nasa.gov/docs/legacy/CMAPSSData.zip`).
- **Dataset loader:** `mammoth/datasets/cmapss.py`
- **Backbones:** `mammoth/backbone/CMAPSScfc.py` (`cmapssmlp`, `cmapsscfc`)
- **Verified:** loader produces 28k+ training windows in ~6s; 1-epoch
  debug-mode smoke test runs end-to-end without errors. A full epoch is
  noticeably slower than TEP/MNIST due to the larger window count and
  30-step CfC unroll per window — budget accordingly when launching full runs.

## 4. AI4I 2020 Predictive Maintenance Dataset (identified, not yet implemented)

- **Citation:** *AI4I 2020 Predictive Maintenance Dataset* [Dataset]. (2020).
  UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
  (introductory paper: S. Matzka, *"Explainable Artificial Intelligence for
  Predictive Maintenance Applications,"* ICAII 2020).
- **Link:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
- **License:** CC BY 4.0
- **Format:** 10,000 rows, 6 real-valued/categorical features (product
  quality variant L/M/H, air temperature, process temperature, rotational
  speed, torque, tool wear) and a machine-failure indicator with 5
  independent failure modes: tool-wear failure (TWF), heat-dissipation
  failure (HDF), power failure (PWF), overstrain failure (OSF), and random
  failure (RNF).
- **Why it's a strong candidate:** unlike SECOM, this dataset has a genuine
  6-class structure (Normal + 5 failure modes) without needing an artificial
  "task-specific class copies" trick — it can use the same even-grouping
  approach as Steel Plates Faults (e.g., 3 tasks of 2 classes each). It is
  also synthetic-but-realistic and widely cited in the predictive-maintenance
  literature, making it easy to contextualize against prior work.
- **Not yet implemented:** left for a follow-up session; the main design
  question is simply which classes to pair per task (analogous to the Steel
  Plates Faults merge/grouping decision), not a preprocessing or missing-data
  problem like SECOM.

## 5. CWRU Bearing Fault Dataset — not added

The official host, Case Western Reserve University's Bearing Data Center
(`engineering.case.edu/bearingdatacenter`), currently **redirects through an
ad-tracking domain** (`match.adsrvr.org`) instead of serving the dataset
page directly. This could be a stale/expired institutional redirect or a
hijacked link chain; either way, it is not something to embed in an
automated download script or cite as a data source without independent
verification through a different channel (e.g., contacting the maintainers
directly, or locating an official DOI-backed mirror). No download script or
loader was added for this dataset. If a verified, official source is found
later, it remains a strong candidate (4-class vibration-signal fault
classification, well known in the PHM literature) for a third industrial
benchmark.
