# MUSCO – Educational Filters & Classroom Tools

**From corpus to classroom.**  
This repository contains code and resources for MUSCO’s educational filters and interactive access to children’s song repertoire. It includes:
- `src/educationalfilters`: the installable Python package (filters, pipeline, utils)
- `notebooks/`: analysis and demo notebooks
- `scripts/`: small command-line helpers and utilities
- `data/` and `exports/`: local inputs/outputs (kept out of version control)
- `assets/`: project visuals and static files

MUSCO is an open-source digital music platform with multiple corpora and support for exploration by metadata, musical features (key, time signature, tempo, ambitus), educational filters (e.g. VRF1/2, IF1/2, RF1–4), and rhythmic or melodic patterns (n-grams). [ICCCM Poster](ICCCM_25_POSTER.pdf)

## Codebase
- Python package under `src/educationalfilters` (filters, pipeline, utilities)  
- Jupyter notebooks for demos on filter functionality  
- Filters are also available to use directly on the MUSCO platform  

## Corpora
- *Ciciban*: 123 digitised children’s songs (JSONs, scores available [here](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/SINZFK). 
- *SLP*: 402 Slovenian folk songs (CSV, scores available [here](https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/SINZFK). 
- **⚠️ Scores of both corpora to be saved in designated folder or code should be altered.⚠️**  

## Repository Structure

```
MUSCO/
├─ assets/                 # Figures, posters, static files
├─ data/                   # Local inputs (ignored by git)
├─ exports/                # Generated outputs (ignored by git)
├─ notebooks/              # Jupyter notebooks
├─ scripts/                # Utility scripts
├─ scores/                 # Where Ciciban and SLP scores should be stored
├─ src/
│  └─ educationalfilters/  # Installable package (filters, pipeline, utils)
├─ .github/workflows/      # CI configuration
├─ .gitignore
├─ LICENSE
└─ README.md
```

## Installation

Create and activate a virtual environment, then install the package in editable mode:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -U pip
pip install -e .
```

Verify the installation:

```bash
python -c "import educationalfilters; print('Educational filters ready')"
```

Alternatively, install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

### 0) Make project modules importable in a notebook/script
```python
import sys
from pathlib import Path

# add project /src and /scripts to sys.path (portable)
p = Path.cwd().resolve()
while p != p.parent and not (p / "src").exists():
    p = p.parent
sys.path.insert(0, str((p / "src").resolve()))
sys.path.insert(0, str((p / "scripts").resolve()))
```
Install basic requirements: 
```bash
pip install -r requirements.txt
```

### 1) Imports used in the notebook

```bash
from educationalfilters import prepare_df, save_load, filter_df
from educationalfilters import dataset_conversion as dc, rfilters
from scripts import plot
import pandas as pd
```
### 2) Imports used in the notebook

```python
REPO_ROOT = p
RAW_DIR    = (REPO_ROOT / "data").resolve()       # where raw inputs live
SCORES_DIR = (REPO_ROOT / "scores").resolve()     # where PDFs/MusicXML live
PROCESSED  = (REPO_ROOT / "data/processed").resolve()
EXPORTS    = (REPO_ROOT / "exports").resolve()
EXPORTS.mkdir(parents=True, exist_ok=True)
```
### 3) Load corpora exactly like the notebook

```python
# Ciciban JSONs
ciciban_json_dir = (RAW_DIR / "ciciban_jsons").resolve()
ciciban_df = prepare_df.convert_jsons_to_df(ciciban_json_dir)

# SLP CSV
slp_csv = (RAW_DIR / "slp_df.csv").resolve()
slp_df  = prepare_df.prepare_slp(slp_csv)

# Upgrade / harmonise schemas
ciciban_df = prepare_df.df_upgrade(ciciban_df)
slp_df     = prepare_df.df_upgrade(slp_df)
```

### 3) Load corpora exactly like the notebook

```python
# Ciciban JSONs
ciciban_json_dir = (RAW_DIR / "ciciban_jsons").resolve()
ciciban_df = prepare_df.convert_jsons_to_df(ciciban_json_dir)

# SLP CSV
slp_csv = (RAW_DIR / "slp_df.csv").resolve()
slp_df  = prepare_df.prepare_slp(slp_csv)

# Upgrade / harmonise schemas
ciciban_df = prepare_df.df_upgrade(ciciban_df)
slp_df     = prepare_df.df_upgrade(slp_df)
```
### 4) Prepare melodic intervals & merge (dataset_conversion)
```python
ciciban_int = dc.prepare_melodic_intervals(ciciban_df)
slp_int     = dc.prepare_melodic_intervals(slp_df)
merged_df = dc.process_and_merge_dfs(ciciban_int, slp_int)
```
### 5) (Optional) light cleaning like in the notebook

```python
# ensure ambitus/time_signature fields are consistent
for col in ("ambitus_min", "ambitus_max"):
    if col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna("")

if "time_signature" in merged_df.columns:
    merged_df["time_signature"] = merged_df["time_signature"].astype(str)
```

### 6) Rhythm labels (rfilters)
```python
# Provide the rhythm mapping exactly as your notebook defines it
rhythm_mapping = {}  # <-- replace with the mapping used in the notebook
merged_df = rfilters.compute_rhythm_labels(merged_df, rhythm_mapping)
```

### 7) Educational filters summary (filter_df)

```python
# Newer path:
summary_clean = filter_df.prepare_all_filters_clean(merged_df)

# If your notebook still calls the legacy name:
# summary_clean = filter_df.preschool_filter(merged_df)
```
### 8) Plot (scripts/plot.py)

```python
COLORS  = {...}   # same as notebook
HATCHES = {...}   # same as notebook
group1  = [...]   # e.g., ["VRF1","VRF2","IF1","IF2"]
group2  = [...]   # e.g., ["RF1","RF2","RF3","RF4"]

plot.plot_filters(summary_clean, COLORS, HATCHES, group1, group2,
                  save_path=str(EXPORTS / "filters_summary.png"))
```

## Licence

- **Code**: MIT (see `LICENSE`)  
- **Data/exports**: follow each corpus’ licence (for example, CC BY-SA or CC BY-NC-SA as applicable). Add a short README in `data/` or `exports/` to state the licence for files you place there.

## Citation

If you use this work in academic contexts, please cite MUSCO and the educational-filters study or poster as appropriate.

## License

See [LICENSE.md](LICENSE.md) for details.

- Code: MIT License (free use, modification, distribution).  
- SLP dataset: CC BY-NC-SA 4.0.  
- Ciciban dataset: CC BY-SA 4.0 (with specific terms).  
