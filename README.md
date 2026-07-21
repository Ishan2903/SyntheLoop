# SyntheLoop
LLM-guided closed-loop pipeline for synthetic tabular data that generates, evaluates, and iteratively refines data quality automatically
# SyntheLoop

**An LLM-guided closed-loop pipeline that generates, evaluates, and iteratively refines synthetic tabular data until it meets quality, privacy, and utility thresholds.**

SyntheLoop takes a real-world tabular dataset, analyzes it, plans a generation strategy with an LLM, trains a CTGAN model, evaluates the synthetic output on multiple axes, and automatically refines the configuration — repeating until the data is good enough or a max iteration count is hit. No manual tuning required.

---

## How it works

```
Input CSV
   │
   ▼
┌─────────────┐      ┌──────────────┐         ┌───────────────┐
│  EDA Module │────▶|  LLM Planner  │────▶   │CTGAN Generator│
└─────────────┘      └──────────────┘         └───────┬───────┘
                                                   │
                                                   ▼
┌───────────────┐     ┌───────────────┐      ┌─────────────┐
│ Refined Config│◀────│  LLM Evaluator │◀───│Quality Metrics│
└──────┬────────┘     └───────────────┘      └─────────────┘
       │
       └──── loop until thresholds met / max iterations ────┐
                                                              │
                                                              ▼
                                          Final synthetic dataset
                                          + quality report
                                          + audit trail
```

1. **EDA** — infers column types, distributions, correlations, missing values, and class balance.
2. **LLM Planner** — reads the grounded EDA summary and proposes a CTGAN configuration (epochs, batch size, categorical columns, generator/discriminator dims).
3. **Generator** — trains CTGAN and samples a synthetic dataset with the same schema.
4. **Evaluator** — computes statistical fidelity (KS test), correlation preservation, class balance (JS divergence), privacy risk (distance to closest record), and ML utility (TSTR vs. TRTR AUC).
5. **LLM Feedback** — reads the grounded metrics and proposes specific configuration adjustments.
6. **Loop** — repeats steps 3–5 until thresholds are met or `max_iterations` is reached, logging every iteration to an audit trail.

## Features

- Fully automated generate → evaluate → refine loop, no manual hyperparameter tuning
- Multi-metric evaluation: fidelity, correlation, class balance, privacy, downstream ML utility
- LLM reasoning grounded strictly in computed statistics (not free recall) to limit hallucinated recommendations
- Full audit trail of every iteration's config, metrics, and LLM feedback
- Local web UI for upload, configuration, live progress, and downloads
- Runs entirely locally except for LLM API calls

## Tech stack

- **Backend:** Python, FastAPI
- **Generative model:** CTGAN (via `ctgan` / `sdv`)
- **EDA & metrics:** pandas, NumPy, SciPy, scikit-learn
- **LLM integration:** Anthropic Claude API (structured JSON prompting)
- **Frontend:** Streamlit (or minimal HTML/JS client)

## Getting started

### Prerequisites
- Python 3.10+
- An Anthropic API key (or other supported LLM provider key)

### Installation
```bash
git clone https://github.com/<your-username>/syntheloop.git
cd syntheloop
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration
Create a `.env` file in the project root:
```
ANTHROPIC_API_KEY=your_key_here
```

### Run locally
```bash
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```
Then open the Streamlit URL shown in your terminal, upload a CSV, set your target column and quality thresholds, and start a run.

## Project structure

See [`STRUCTURE.md`](./STRUCTURE.md) for the full annotated file layout.

## Example datasets

Demo runs are validated against public datasets such as the UCI Adult Income dataset and a Kaggle customer churn dataset. Sample inputs live in `data/samples/`.

## Roadmap

- Support additional generators (TVAE, Gaussian Copula) beyond CTGAN
- Multi-provider LLM support (OpenAI, local models)
- Batch/scheduled runs
- Dockerized deployment

## License

MIT — see [LICENSE](./LICENSE).

## Acknowledgments

- Xu, L. et al., *Modeling Tabular Data using Conditional GAN*, NeurIPS 2019 (CTGAN)
- [SDV — Synthetic Data Vault](https://sdv.dev/)
