# SyntheLoop — Full Implementation Methodology

**Purpose of this document:** a complete, unambiguous build spec. Every module below lists its exact file path, purpose, function/class signatures, inputs/outputs, data contracts, and a definition of done. Build in the order given in Section 6 — each module depends only on modules built before it.

---

## 1. Tech Stack & Exact Dependencies

```
pandas
numpy
scipy
scikit-learn
ctgan
fastapi
uvicorn
streamlit
python-dotenv
groq
pydantic
```

Python 3.10+. Install with `pip install -r requirements.txt`. All LLM calls use the Groq Python SDK (`groq` package, OpenAI-compatible chat completions interface) against `GROQ_API_KEY` read from environment. Default model: `llama-3.3-70b-versatile` (strong at structured JSON output; swap to a different Groq-hosted model in `.env` if needed).

---

## 2. Environment & Configuration

### `.env.example` (copy to `.env`, fill in, never commit `.env`)
```
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
MAX_ITERATIONS_DEFAULT=5
```

### `backend/config.py`
```python
from pydantic import BaseModel
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    max_iterations_default: int = int(os.getenv("MAX_ITERATIONS_DEFAULT", 5))
    output_dir: str = "outputs"
    upload_dir: str = "data/uploads"

    # Default quality thresholds — overridable per run via API/UI
    default_thresholds: dict = {
        "ks_stat_max": 0.15,          # lower is better; per-column, take mean
        "corr_diff_max": 0.20,        # Frobenius-norm difference, normalized
        "js_divergence_max": 0.10,    # class balance divergence
        "dcr_min_percentile": 5,      # privacy: 5th percentile of nearest-neighbor distance must exceed a safe floor
        "utility_auc_drop_max": 0.10  # max allowed drop of TSTR AUC vs TRTR AUC
    }

settings = Settings()
```

**Definition of done:** `from backend.config import settings` works; all values load from `.env`; missing `.env` falls back to defaults without crashing.

---

## 3. File Structure (canonical — do not deviate)

```
syntheloop/
├── backend/
│   ├── main.py
│   ├── config.py
│   ├── eda/{__init__.py, analyzer.py}
│   ├── planner/{__init__.py, llm_planner.py, prompts.py}
│   ├── generator/{__init__.py, ctgan_wrapper.py}
│   ├── evaluator/{__init__.py, metrics.py, llm_evaluator.py, prompts.py}
│   ├── pipeline/{__init__.py, feedback_loop.py, audit_trail.py}
│   └── reports/{__init__.py, report_builder.py}
├── frontend/app.py
├── data/samples/
├── outputs/
├── tests/{test_eda.py, test_generator.py, test_evaluator.py, test_pipeline.py}
└── docs/
```

---

## 4. Data Contracts (JSON schemas — all LLM I/O must conform to these exactly)

### 4.1 EDA Summary (output of `backend/eda/analyzer.py`)
```json
{
  "n_rows": 32561,
  "n_cols": 15,
  "target_column": "income",
  "columns": {
    "<col_name>": {
      "dtype": "int64 | float64 | object",
      "type": "continuous | categorical",
      "missing_pct": 0.0,
      "n_unique": 42,
      "mean": 38.5, "std": 13.6, "skew": 0.5, "kurtosis": -0.2, "min": 17, "max": 90,
      "top_categories": {"cat_a": 0.4, "cat_b": 0.3},
      "balance_ratio": 0.75
    }
  },
  "correlation_matrix": {"<col_a>": {"<col_b>": 0.34}},
  "class_balance": {"<class_label>": 0.24},
  "categorical_columns": ["workclass", "education", "..."]
}
```
Numeric-only fields appear for `type: continuous`; categorical-only fields appear for `type: categorical`. This exact structure is what gets serialized and sent to the LLM planner — never send raw row data.

### 4.2 Generation Config (LLM Planner output → Generator input)
```json
{
  "categorical_columns": ["workclass", "education", "marital_status"],
  "epochs": 150,
  "batch_size": 500,
  "generator_dim": [256, 256],
  "discriminator_dim": [256, 256],
  "pac": 10,
  "reasoning": "one-sentence justification, for the audit trail, not used programmatically"
}
```

### 4.3 Metrics Output (Evaluator → LLM Evaluator input)
```json
{
  "iteration": 2,
  "per_column_ks": {"<col_name>": 0.12},
  "correlation_diff_frobenius": 0.18,
  "class_balance_js_divergence": 0.05,
  "privacy_dcr_5th_percentile": 0.31,
  "utility": {"trtr_auc": 0.87, "tstr_auc": 0.81, "auc_drop": 0.06},
  "thresholds": { "...copied from Settings.default_thresholds or run override..." },
  "passed": {"ks": true, "correlation": false, "balance": true, "privacy": true, "utility": true},
  "overall_passed": false
}
```

### 4.4 LLM Feedback (LLM Evaluator output → Refinement input)
```json
{
  "weak_areas": ["correlation"],
  "weak_columns": ["capital_gain", "hours_per_week"],
  "diagnosis": "one to three sentences, grounded only in the metrics JSON provided, no invented column names",
  "config_adjustments": {
    "epochs": 250,
    "generator_dim": [512, 256],
    "discriminator_dim": [512, 256]
  },
  "stop_recommended": false
}
```
`config_adjustments` contains **only the keys being changed** — the refinement step merges these into the previous config, it does not replace it wholesale.

### 4.5 Audit Trail Entry (one per iteration, written by `pipeline/audit_trail.py`)
```json
{
  "run_id": "uuid",
  "iteration": 2,
  "timestamp": "ISO-8601",
  "config_used": { "...4.2 schema..." },
  "metrics": { "...4.3 schema..." },
  "feedback": { "...4.4 schema..." },
  "action_taken": "continued | stopped_threshold_met | stopped_max_iterations | failed"
}
```

---

## 5. Module-by-Module Implementation Spec

### 5.1 `backend/eda/analyzer.py`

```python
class EDAAnalyzer:
    def __init__(self, df: pd.DataFrame, target_col: str | None = None): ...
    def analyze(self) -> dict:
        """Returns a dict matching the Data Contract 4.1 schema exactly.
        Must handle: all-numeric datasets, all-categorical datasets,
        a single-class target column, and columns that are 100% missing.
        Never raises on well-formed CSVs; raises ValueError with a clear
        message on malformed input (e.g. zero columns)."""
```

**Rules:**
- A column is `continuous` if `pd.api.types.is_numeric_dtype` is true AND `n_unique > 20` (tune threshold); otherwise `categorical`. Document this rule inline — it determines what gets sent to CTGAN as discrete vs. continuous.
- `correlation_matrix` computed only over numeric columns; omit the key entirely if fewer than 2 numeric columns exist.
- `class_balance` computed only if `target_col` is provided and exists in `df.columns`.

**Definition of done:** unit test in `tests/test_eda.py` runs `EDAAnalyzer` against (a) a mixed-type sample CSV, (b) an all-numeric CSV, (c) a CSV with a fully-missing column, and asserts the output matches the 4.1 schema shape (keys present, correct types) in all three cases.

---

### 5.2 `backend/generator/ctgan_wrapper.py`

```python
class SyntheticGenerator:
    def __init__(self, config: dict): ...  # matches Data Contract 4.2
    def fit(self, real_df: pd.DataFrame) -> None:
        """Trains CTGAN using self.config['categorical_columns'], epochs,
        batch_size, generator_dim, discriminator_dim. Must run on CPU
        without requiring CUDA (GPU used automatically if available via
        the ctgan library's own device detection — do not hardcode device)."""
    def sample(self, n_rows: int) -> pd.DataFrame:
        """Returns a synthetic DataFrame with identical columns/dtypes to
        the real_df used in fit()."""
```

**Rules:**
- Must validate that every name in `config['categorical_columns']` actually exists in `real_df.columns` before calling CTGAN — raise `ValueError` listing the invalid names if not (this is the primary defense against LLM hallucination per NFR-7: the planner might invent a column name, and this check catches it before wasting a training run).
- Wrap `ctgan.CTGAN(...)` — do not reimplement GAN training.

**Definition of done:** `tests/test_generator.py` fits on a small (≤500 row) sample dataset, samples 100 rows, and asserts output shape/dtypes match input.

---

### 5.3 `backend/evaluator/metrics.py`

```python
def compute_ks_per_column(real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    """scipy.stats.ks_2samp per numeric column. Returns {col: ks_statistic}."""

def compute_correlation_diff(real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: list[str]) -> float:
    """Frobenius norm of (real_corr_matrix - synth_corr_matrix), normalized
    by matrix size. Returns a single float."""

def compute_class_balance_js(real: pd.Series, synth: pd.Series) -> float:
    """scipy.spatial.distance.jensenshannon on the two class-frequency
    distributions (align categories, fill missing categories with 0)."""

def compute_privacy_dcr(real: pd.DataFrame, synth: pd.DataFrame, numeric_cols: list[str]) -> float:
    """Distance to Closest Record: for each synthetic row, nearest-neighbor
    Euclidean distance (on normalized numeric columns) to any real row.
    Return the 5th percentile of that distribution — a low value means
    some synthetic rows are suspiciously close to real ones (privacy risk)."""

def compute_ml_utility(real: pd.DataFrame, synth: pd.DataFrame, target_col: str) -> dict:
    """TRTR: train RandomForestClassifier on real (80/20 split), test on
    real holdout -> auc_trtr.
    TSTR: train RandomForestClassifier on synth, test on the SAME real
    holdout -> auc_tstr.
    Returns {"trtr_auc": ..., "tstr_auc": ..., "auc_drop": trtr_auc - tstr_auc}.
    Use sklearn LabelEncoder for categorical target; roc_auc_score with
    multi_class='ovr' if more than 2 classes."""

def evaluate_all(real, synth, eda_summary, thresholds, iteration) -> dict:
    """Calls all five functions above, assembles Data Contract 4.3 output,
    including per-metric pass/fail against thresholds and overall_passed
    (AND of all five)."""
```

**Definition of done:** `tests/test_evaluator.py` runs `evaluate_all` on a real/synthetic pair from `test_generator.py`'s output, asserts every key in Data Contract 4.3 is present with correct types, and that `overall_passed` is a bool.

---

### 5.4 `backend/planner/prompts.py`

```python
PLANNER_SYSTEM_PROMPT = """You are a configuration planner for a CTGAN-based
synthetic data generator. You will be given a JSON summary of a dataset's
exploratory data analysis. Respond with ONLY a JSON object matching this
exact schema, no prose, no markdown fences:
{
  "categorical_columns": [list of column names from the input, exactly as spelled],
  "epochs": integer between 50 and 500,
  "batch_size": integer, multiple of 10, between 50 and 1000,
  "generator_dim": [int, int],
  "discriminator_dim": [int, int],
  "pac": integer between 1 and 20,
  "reasoning": "one sentence"
}
Rules:
- categorical_columns MUST be a subset of the "categorical_columns" list
  already present in the input JSON. Never invent a column name not present
  in the input.
- Base epochs/batch_size on n_rows: smaller datasets (<5000 rows) need
  fewer epochs to avoid overfitting; larger datasets can use more.
- If this is a refinement call (input includes prior feedback), adjust
  only the parameters the feedback flags as weak."""

def build_planner_prompt(eda_summary: dict, prior_feedback: dict | None = None) -> str:
    """Serializes eda_summary (and prior_feedback if present) to a compact
    JSON string and returns the full user-turn prompt text. Never includes
    raw dataset rows — only the EDA summary JSON."""
```

### 5.5 `backend/planner/llm_planner.py`

```python
class LLMPlanner:
    def __init__(self, client: groq.Groq, model: str): ...
    def plan(self, eda_summary: dict, prior_feedback: dict | None = None) -> dict:
        """Calls the Groq API (client.chat.completions.create, OpenAI-
        compatible interface) with PLANNER_SYSTEM_PROMPT as the system
        message and the built user prompt as the user message. Pass
        response_format={"type": "json_object"} if the selected Groq model
        supports it, to reduce the chance of prose/markdown wrapping the
        JSON. Parses the response as JSON (strip markdown fences if
        present, as a fallback for models that don't honor json_object
        mode). Validates against Data Contract 4.2. On parse failure or
        schema mismatch: retry once with an added instruction emphasizing
        JSON-only output; on second failure, fall back to a hardcoded
        default config and log a warning (never crash the pipeline on a
        malformed LLM response)."""
```

**Definition of done:** given a hand-written EDA summary fixture, `plan()` returns a dict matching Data Contract 4.2, with `categorical_columns` validated as a subset of the input's categorical columns.

---

### 5.6 `backend/evaluator/prompts.py` and `backend/evaluator/llm_evaluator.py`

Same pattern as 5.4/5.5, but:
- System prompt instructs the LLM to output Data Contract 4.4 JSON exactly.
- Explicitly instructs: *"Base your diagnosis and weak_columns list only on the metrics JSON provided. Do not reference columns not present in per_column_ks."*
- `LLMEvaluator.evaluate_feedback(metrics: dict) -> dict` — same retry/fallback behavior as the planner (on repeated failure, fall back to a rule-based default: flag whichever single metric is furthest from its threshold, recommend +50 epochs).

**Definition of done:** given a hand-written metrics fixture (Data Contract 4.3) with `correlation: false`, `evaluate_feedback()` returns feedback whose `weak_areas` includes `"correlation"`.

---

### 5.7 `backend/pipeline/audit_trail.py`

```python
class AuditTrail:
    def __init__(self, run_id: str, output_dir: str): ...
    def log_iteration(self, entry: dict) -> None:
        """Appends one Data Contract 4.5 entry to
        {output_dir}/{run_id}/audit_trail.json (a JSON array, load-append-
        rewrite is fine at this scale — no database needed)."""
    def get_full_trail(self) -> list[dict]: ...
```

---

### 5.8 `backend/pipeline/feedback_loop.py`

```python
class FeedbackLoop:
    def __init__(self, real_df, target_col, thresholds, max_iterations,
                 planner: LLMPlanner, evaluator_metrics_fn, llm_evaluator: LLMEvaluator,
                 audit_trail: AuditTrail): ...

    def run(self) -> dict:
        """
        1. eda_summary = EDAAnalyzer(real_df, target_col).analyze()
        2. prior_feedback = None
        3. for iteration in range(1, max_iterations + 1):
             config = planner.plan(eda_summary, prior_feedback)
             generator = SyntheticGenerator(config); generator.fit(real_df)
             synth_df = generator.sample(len(real_df))
             metrics = evaluate_all(real_df, synth_df, eda_summary, thresholds, iteration)
             feedback = llm_evaluator.evaluate_feedback(metrics)
             audit_trail.log_iteration({... Data Contract 4.5 ...})
             if metrics['overall_passed'] or feedback.get('stop_recommended'):
                 return {"status": "completed_threshold_met", "final_synth": synth_df,
                         "final_metrics": metrics, "iterations_used": iteration}
             prior_feedback = feedback  # merge config_adjustments into next planner call
        return {"status": "completed_max_iterations", "final_synth": synth_df,
                "final_metrics": metrics, "iterations_used": max_iterations}
        """
```

**Rules:**
- Never mutate `config` in place across iterations — always construct a fresh config dict per iteration from planner output, so the audit trail's `config_used` is an accurate snapshot.
- Any exception inside the loop (LLM API error, CTGAN training failure) must be caught, logged to the audit trail with `action_taken: "failed"`, and re-raised or returned as a `status: "failed"` result — never leave the loop in an undefined state.

**Definition of done:** `tests/test_pipeline.py` runs `FeedbackLoop.run()` end-to-end on a small sample dataset with `max_iterations=2`, asserts it returns a dict with `status` in the expected set and `final_synth` is a non-empty DataFrame.

---

### 5.9 `backend/reports/report_builder.py`

```python
def build_report(run_id: str, eda_summary: dict, audit_trail: list[dict], final_metrics: dict) -> str:
    """Renders an HTML report (Jinja2 or plain f-strings) summarizing:
    - Dataset overview (from eda_summary)
    - Metric trend across iterations (table: iteration -> all 5 metric values)
    - Final pass/fail per metric
    - Full audit trail (collapsible per-iteration detail: config used, feedback text)
    Returns the HTML string; caller writes it to
    outputs/{run_id}/report.html"""
```

---

### 5.10 `backend/main.py` (FastAPI)

Endpoints:
- `POST /upload` — accepts CSV, validates via a lightweight check, saves to `data/uploads/{run_id}.csv`, returns `run_id` and `EDAAnalyzer(...).analyze()` output.
- `POST /runs/{run_id}/start` — body: `{target_col, thresholds (optional, else defaults), max_iterations}`. Instantiates `FeedbackLoop`, runs it (consider running in a background task so the endpoint returns immediately and the frontend polls status).
- `GET /runs/{run_id}/status` — returns current iteration, latest metrics, whether still running.
- `GET /runs/{run_id}/download/{artifact}` — `artifact` in `{dataset, report, audit_trail}`, streams the corresponding file from `outputs/{run_id}/`.

---

### 5.11 `frontend/app.py` (Streamlit)

Sections, top to bottom:
1. File uploader → calls `POST /upload` → displays EDA summary (st.dataframe/st.json).
2. Form: target column selectbox (populated from EDA summary columns), threshold sliders (5, one per metric, defaulting to `Settings.default_thresholds`), max_iterations number input.
3. "Start Run" button → calls `POST /runs/{run_id}/start`.
4. Polling loop (`st.empty()` placeholder updated on a timer, or `st.rerun()` pattern) showing current iteration and live metric values from `GET /runs/{run_id}/status`.
5. On completion: three `st.download_button`s wired to the download endpoint for dataset, report, audit trail.

---

## 6. Build Order (strict — do not skip ahead)

1. `backend/config.py`
2. `backend/eda/analyzer.py` + `tests/test_eda.py`
3. `backend/generator/ctgan_wrapper.py` + `tests/test_generator.py` (manual/hardcoded config at this stage — no planner yet)
4. `backend/evaluator/metrics.py` + `tests/test_evaluator.py`
5. `backend/planner/prompts.py` + `backend/planner/llm_planner.py`
6. `backend/evaluator/prompts.py` + `backend/evaluator/llm_evaluator.py`
7. `backend/pipeline/audit_trail.py`
8. `backend/pipeline/feedback_loop.py` + `tests/test_pipeline.py` — first point where the full loop exists and must be tested end-to-end
9. `backend/reports/report_builder.py`
10. `backend/main.py`
11. `frontend/app.py`
12. Generalize: run against 2-3 additional sample datasets in `data/samples/`, fix whatever edge cases surface
13. Polish: error handling audit, logging cleanup

## 7. Out of Scope (do not build these — guardrails against over-engineering)

- No database — flat JSON/CSV files under `outputs/{run_id}/` are the persistence layer.
- No authentication/multi-user support — single local user.
- No generators other than CTGAN in this version.
- No distributed/async task queue (Celery, etc.) — a simple FastAPI background task is sufficient at this scale.
- No support for non-tabular data (text, images, time series).

## 8. Global Error-Handling Requirements (apply everywhere an LLM or CTGAN call happens)

- Every LLM call: wrap in try/except, retry once on JSON-parse failure, fall back to a hardcoded default on second failure — never let a malformed LLM response crash the pipeline.
- Groq-specific: not every Groq-hosted model supports `response_format={"type": "json_object"}` — confirm support for whichever `GROQ_MODEL` is configured before relying on it, and always keep the markdown-fence-stripping fallback regardless. Groq's free/dev tier also enforces requests-per-minute rate limits; wrap calls with a short backoff-and-retry (e.g. 2-3 attempts with increasing delay) on HTTP 429 responses, separate from the JSON-parse retry above.
- Every CTGAN fit/sample call: wrap in try/except; on failure, log to audit trail with `action_taken: "failed"` and surface a clear error to the API/UI rather than a raw stack trace.
- Dataset validation (`EDAAnalyzer` construction or an explicit pre-check in `main.py`'s `/upload`): reject empty files, files with zero numeric AND zero categorical usable columns, and files with a single row, with a specific error message per case — never a generic "500 Internal Server Error."
