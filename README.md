# Educational Filters for Music Corpora

This repository implements a framework for creating **educational filters** (VRF1, VRF2, IF1, IF2, RF1–RF4), designed on the basis of a literature review in music education. The goal is to provide computationally supported tools for filtering, analysing, and exploring music corpora in educational contexts.

## Features

- **Educational filters**  
  - VRF1, VRF2 (Vocal range filters)  
  - IF1, IF2 (Interval-related filters)  
  - RF1–RF4 (Rhythm-related filters)  

- **Codebase**  
  - Implemented in Python.  
  - Dataset conversion functions for applying filters and analyses.  
  - Pattern matching module, adapted and extended from the [XX Project](https://...).  

- **Datasets**  
  - `ciciban`: 123 digitised children’s songs (scores and JSON representations).  
  - `slp`: 402 Slovenian folk songs, re-uploaded from the other project.  

- **Jupyter Notebooks**  
  - Interactive notebooks to launch filters and analyses more easily.  

## Repository Structure

```
src/educationalfilters/      # Core Python code (filters, dataset conversion, pattern matching)
notebooks/              # Jupyter notebooks for demonstrations
scripts/                # Utility scripts
scores/                 # Ciciban and SLP scores
data/                   # Processed data and JSONs (not versioned)
exports/                # Generated results
assets/                 # Figures, documents
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/vnborsan/MUSCO.git
cd MUSCO
```

Install dependencies (recommended: in a virtual environment):

```bash
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

## Usage

Example: run dataset conversion with filters.

```bash
python -m musco.dataset_conversion
```

Example: launch Jupyter Notebook for interactive analysis.

```bash
jupyter notebook notebooks/musco_edu.ipynb
```

## License

See [LICENSE.md](LICENSE.md) for details.

- Code: MIT License (free use, modification, distribution).  
- SLP dataset: CC BY-NC-SA 4.0.  
- Ciciban dataset: CC BY-SA 4.0 (with specific terms).  
