# Lexical cues in LLM persuasion and in web search sources

Code, word lists and results for the preprint

> Aksoy, O. (2026). *Lexical Cues in LLM Persuasion and in Web Search Sources: A Reanalysis of Two Public Benchmarks.* Preprints.org.

The analysis uses two public datasets:

- **Anthropic/persuasion**: 3,939 ratings of 1,313 arguments (human and five Claude models), https://huggingface.co/datasets/Anthropic/persuasion
- **GEO-bench (test split)**: 1,000 queries with Google top-five pages, https://huggingface.co/datasets/GEO-Optim/geo-bench

The datasets are not redistributed here; `src/download.py` fetches them.

## Reproduce

Python 3.12 was used.

```bash
pip install -r requirements.txt
python src/download.py      # ~46 MB into data/
python src/analysis.py      # results/results.json and per-argument/per-page CSV files
python src/figures.py       # figures/*.pdf, figures/*.png, figures/graphical_abstract.png
python src/tables.py        # tables/*.tex
```

`analysis.py` takes about 20 seconds on a laptop. The random seed is fixed (`SEED` in `src/analysis.py`).

## Layout

| Path | Content |
|---|---|
| `src/features.py` | Word lists and text features (per 100 words) |
| `src/analysis.py` | All statistics for Corpus A and Corpus B |
| `src/figures.py`, `src/tables.py` | Figures and LaTeX tables from `results/results.json` |
| `results/` | `results.json` (every number in the paper), `corpus_a_arguments.csv`, `corpus_b_documents.csv` |

Figures and tables are not stored in the repository; the two scripts recreate them from `results/results.json`.

## Notes on the method

- Each argument in Anthropic/persuasion was rated by three workers, so comparisons of belief change use argument means; confidence intervals resample arguments.
- Query intent in GEO-bench: `transactional` tag → Transactional; else `opinion` or `debate` tag → Opinion/Debate; else Informational.
- Word lists match the listed word or the word plus "s"/"es"; the lists are pairwise disjoint (checked at run time).

## Licence

Code: MIT (see `LICENSE`). The datasets keep their original licences.
