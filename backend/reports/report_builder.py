"""SyntheLoop — HTML Report Builder Module.

Renders an executive, self-contained HTML report summarizing:
- Run metadata and executive quality scorecard
- Dataset overview (EDA summary)
- Metric trends across all iterations
- Collapsible iteration-by-iteration audit trail
"""

from datetime import datetime, timezone
import html
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _format_float(val: Any, precision: int = 4) -> str:
    """Safely formats a float or returns 'N/A'."""
    if val is None:
        return "N/A"
    try:
        f = float(val)
        return f"{f:.{precision}f}"
    except (ValueError, TypeError):
        return str(val)


def build_report(
    run_id: str,
    eda_summary: dict[str, Any] | None,
    audit_trail: list[dict[str, Any]] | None,
    final_metrics: dict[str, Any] | None,
) -> str:
    """Renders a self-contained HTML report.

    Args:
        run_id: Unique identifier for the pipeline run.
        eda_summary: Output dictionary from EDAAnalyzer.analyze() (Data Contract 4.1).
        audit_trail: List of iteration records (Data Contract 4.5).
        final_metrics: Final evaluation metrics dictionary (Data Contract 4.3).

    Returns:
        str: Fully-formed HTML document string.
    """
    eda_summary = eda_summary or {}
    audit_trail = audit_trail or []
    final_metrics = final_metrics or {}

    # Metadata
    clean_run_id = html.escape(str(run_id))
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_iterations = len(audit_trail)
    overall_passed = bool(final_metrics.get("overall_passed", False))

    status_badge = (
        '<span class="badge badge-pass">PASSED</span>'
        if overall_passed
        else '<span class="badge badge-fail">THRESHOLD NOT MET</span>'
    )

    # Metrics extraction
    passed_flags = final_metrics.get("passed", {})
    thresholds = final_metrics.get("thresholds", {})

    # 1. KS
    ks_per_col = final_metrics.get("per_column_ks", {})
    ks_vals = [v for v in ks_per_col.values() if isinstance(v, (int, float))]
    mean_ks = sum(ks_vals) / len(ks_vals) if ks_vals else None
    ks_passed = passed_flags.get("ks", False)
    ks_thresh = thresholds.get("ks_stat_max", 0.15)

    # 2. Correlation
    corr_diff = final_metrics.get("correlation_diff_frobenius")
    corr_passed = passed_flags.get("correlation", False)
    corr_thresh = thresholds.get("corr_diff_max", 0.20)

    # 3. Class Balance
    js_div = final_metrics.get("class_balance_js_divergence")
    js_passed = passed_flags.get("balance", True)
    js_thresh = thresholds.get("js_divergence_max", 0.10)

    # 4. Privacy DCR
    dcr_5th = final_metrics.get("privacy_dcr_5th_percentile")
    dcr_passed = passed_flags.get("privacy", True)
    dcr_thresh = thresholds.get("dcr_min_percentile", 5)

    # 5. ML Utility
    utility = final_metrics.get("utility", {})
    trtr_auc = utility.get("trtr_auc")
    tstr_auc = utility.get("tstr_auc")
    auc_drop = utility.get("auc_drop")
    util_passed = passed_flags.get("utility", True)
    util_thresh = thresholds.get("utility_auc_drop_max", 0.10)

    # Dataset overview fields
    n_rows = eda_summary.get("n_rows", "N/A")
    n_cols = eda_summary.get("n_cols", "N/A")
    target_col = eda_summary.get("target_column") or "None"
    columns_info = eda_summary.get("columns", {})

    # Construct columns table rows
    col_rows = []
    for col_name, info in columns_info.items():
        c_name = html.escape(str(col_name))
        c_dtype = html.escape(str(info.get("dtype", "unknown")))
        c_type = html.escape(str(info.get("type", "unknown")))
        c_missing = f"{info.get('missing_pct', 0.0):.1f}%"
        c_unique = str(info.get("n_unique", "N/A"))

        if info.get("type") == "continuous":
            mean_val = _format_float(info.get("mean"), 2)
            std_val = _format_float(info.get("std"), 2)
            c_detail = f"Mean: {mean_val}, Std: {std_val}"
        else:
            top_cats = info.get("top_categories", {})
            cat_str = ", ".join([f"{k}: {v*100:.0f}%" for k, v in list(top_cats.items())[:2]])
            c_detail = cat_str if cat_str else "N/A"

        col_rows.append(
            f"<tr>"
            f"<td><strong>{c_name}</strong></td>"
            f"<td><code>{c_dtype}</code></td>"
            f"<td><span class='badge-pill'>{c_type}</span></td>"
            f"<td>{c_missing}</td>"
            f"<td>{c_unique}</td>"
            f"<td class='muted-text'>{html.escape(c_detail)}</td>"
            f"</tr>"
        )
    cols_table_html = "\n".join(col_rows) if col_rows else "<tr><td colspan='6'>No column data available</td></tr>"

    # Construct iteration trend table rows
    trend_rows = []
    for entry in audit_trail:
        it = entry.get("iteration", "N/A")
        m = entry.get("metrics", {})
        action = entry.get("action_taken", "continued")

        # Metric values for this iteration
        it_ks_dict = m.get("per_column_ks", {})
        it_ks_vals = [v for v in it_ks_dict.values() if isinstance(v, (int, float))]
        it_mean_ks = _format_float(sum(it_ks_vals) / len(it_ks_vals)) if it_ks_vals else "N/A"
        it_corr = _format_float(m.get("correlation_diff_frobenius"))
        it_js = _format_float(m.get("class_balance_js_divergence"))
        it_dcr = _format_float(m.get("privacy_dcr_5th_percentile"))
        it_drop = _format_float(m.get("utility", {}).get("auc_drop"))
        it_passed = m.get("overall_passed", False)
        it_status_badge = (
            "<span class='badge-pill badge-pass-pill'>Pass</span>"
            if it_passed
            else "<span class='badge-pill badge-fail-pill'>Fail</span>"
        )

        trend_rows.append(
            f"<tr>"
            f"<td><strong>#{it}</strong></td>"
            f"<td>{it_mean_ks}</td>"
            f"<td>{it_corr}</td>"
            f"<td>{it_js}</td>"
            f"<td>{it_dcr}</td>"
            f"<td>{it_drop}</td>"
            f"<td>{it_status_badge}</td>"
            f"<td><code>{html.escape(action)}</code></td>"
            f"</tr>"
        )
    trend_table_html = "\n".join(trend_rows) if trend_rows else "<tr><td colspan='8'>No iteration history available</td></tr>"

    # Construct collapsible audit trail cards
    trail_cards = []
    for entry in audit_trail:
        it = entry.get("iteration", "N/A")
        ts = html.escape(str(entry.get("timestamp", "")))
        action = html.escape(str(entry.get("action_taken", "")))
        cfg = entry.get("config_used", {})
        fb = entry.get("feedback") or {}

        epochs = cfg.get("epochs", "N/A")
        batch_size = cfg.get("batch_size", "N/A")
        gen_dim = cfg.get("generator_dim", "N/A")
        disc_dim = cfg.get("discriminator_dim", "N/A")
        reasoning = html.escape(str(cfg.get("reasoning", "None")))

        diagnosis = html.escape(str(fb.get("diagnosis", "No feedback recorded.")))
        weak_areas = ", ".join(fb.get("weak_areas", [])) or "None"
        weak_cols = ", ".join(fb.get("weak_columns", [])) or "None"
        adjustments = json.dumps(fb.get("config_adjustments", {}), indent=2)

        trail_cards.append(
            f"""
            <details class="audit-card">
                <summary class="audit-summary">
                    <span class="audit-title">Iteration #{it}</span>
                    <span class="audit-meta">Action: <code>{action}</code> &bull; {ts}</span>
                </summary>
                <div class="audit-body">
                    <div class="audit-grid">
                        <div class="audit-col">
                            <h4>Generator Configuration</h4>
                            <ul class="kv-list">
                                <li><strong>Epochs:</strong> {epochs}</li>
                                <li><strong>Batch Size:</strong> {batch_size}</li>
                                <li><strong>Architecture:</strong> Gen {gen_dim} / Disc {disc_dim}</li>
                                <li><strong>Reasoning:</strong> <em>{reasoning}</em></li>
                            </ul>
                        </div>
                        <div class="audit-col">
                            <h4>Evaluator Diagnosis</h4>
                            <ul class="kv-list">
                                <li><strong>Weak Areas:</strong> {html.escape(weak_areas)}</li>
                                <li><strong>Weak Columns:</strong> {html.escape(weak_cols)}</li>
                                <li><strong>Diagnosis:</strong> {diagnosis}</li>
                            </ul>
                            <h4>Recommended Adjustments</h4>
                            <pre><code>{html.escape(adjustments)}</code></pre>
                        </div>
                    </div>
                </div>
            </details>
            """
        )
    audit_trail_html = "\n".join(trail_cards) if trail_cards else "<p class='muted-text'>No audit trail logged.</p>"

    # Helper card renderer for scorecard
    def _card(title: str, val_str: str, passed: bool, threshold_str: str, subtext: str) -> str:
        status_cls = "card-pass" if passed else "card-fail"
        badge_cls = "badge-pass-pill" if passed else "badge-fail-pill"
        badge_lbl = "PASS" if passed else "FAIL"
        return f"""
        <div class="metric-card {status_cls}">
            <div class="metric-header">
                <span class="metric-title">{title}</span>
                <span class="badge-pill {badge_cls}">{badge_lbl}</span>
            </div>
            <div class="metric-value">{val_str}</div>
            <div class="metric-subtext">{subtext}</div>
            <div class="metric-threshold">Threshold: {threshold_str}</div>
        </div>
        """

    ks_card = _card(
        "KS Statistic (Mean)",
        _format_float(mean_ks),
        ks_passed,
        f"&le; {_format_float(ks_thresh, 2)}",
        "Empirical distribution similarity",
    )
    corr_card = _card(
        "Correlation Difference",
        _format_float(corr_diff),
        corr_passed,
        f"&le; {_format_float(corr_thresh, 2)}",
        "Frobenius norm correlation drift",
    )
    js_card = _card(
        "Class Balance (JS)",
        _format_float(js_div),
        js_passed,
        f"&le; {_format_float(js_thresh, 2)}",
        "Jensen-Shannon divergence",
    )
    dcr_card = _card(
        "Privacy (DCR 5th %ile)",
        _format_float(dcr_5th),
        dcr_passed,
        f"&ge; {_format_float(dcr_thresh, 2)}",
        "Nearest-neighbor Euclidean floor",
    )
    util_card = _card(
        "ML Utility (AUC Drop)",
        _format_float(auc_drop),
        util_passed,
        f"&le; {_format_float(util_thresh, 2)}",
        f"TRTR: {_format_float(trtr_auc, 3)} | TSTR: {_format_float(tstr_auc, 3)}",
    )

    # Full HTML template
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SyntheLoop Synthesis Report — {clean_run_id}</title>
    <style>
        :root {{
            --bg: #0b0f19;
            --surface: #111827;
            --surface-card: #1f2937;
            --surface-hover: #374151;
            --border: #374151;
            --border-light: #4b5563;
            --text-main: #f9fafb;
            --text-muted: #9ca3af;
            --accent-primary: #6366f1;
            --accent-cyan: #06b6d4;
            --pass: #10b981;
            --pass-bg: rgba(16, 185, 129, 0.12);
            --fail: #f43f5e;
            --fail-bg: rgba(244, 63, 94, 0.12);
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            background-color: var(--bg);
            color: var(--text-main);
            font-family: var(--font-family);
            line-height: 1.5;
            padding: 2.5rem 1.5rem;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        /* Header */
        header {{
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border: 1px solid #312e81;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        }}

        .header-top {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1rem;
        }}

        h1 {{
            font-size: 1.85rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.025em;
        }}

        .header-meta {{
            display: flex;
            gap: 1.5rem;
            flex-wrap: wrap;
            font-size: 0.9rem;
            color: #c7d2fe;
        }}

        .header-meta span strong {{
            color: #ffffff;
        }}

        /* Badges */
        .badge {{
            display: inline-block;
            padding: 0.35rem 0.85rem;
            font-size: 0.85rem;
            font-weight: 700;
            border-radius: 9999px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .badge-pass {{
            background-color: var(--pass);
            color: #ffffff;
        }}

        .badge-fail {{
            background-color: var(--fail);
            color: #ffffff;
        }}

        .badge-pill {{
            display: inline-block;
            padding: 0.2rem 0.55rem;
            font-size: 0.75rem;
            font-weight: 600;
            border-radius: 4px;
            background-color: #374151;
            color: #d1d5db;
        }}

        .badge-pass-pill {{
            background-color: var(--pass-bg);
            color: var(--pass);
            border: 1px solid rgba(16, 185, 129, 0.3);
        }}

        .badge-fail-pill {{
            background-color: var(--fail-bg);
            color: var(--fail);
            border: 1px solid rgba(244, 63, 94, 0.3);
        }}

        /* Sections */
        section {{
            margin-bottom: 2.5rem;
        }}

        h2 {{
            font-size: 1.35rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #e5e7eb;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        /* Scorecard Grid */
        .scorecard-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 1rem;
        }}

        .metric-card {{
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .metric-card.card-pass {{
            border-top: 4px solid var(--pass);
        }}

        .metric-card.card-fail {{
            border-top: 4px solid var(--fail);
        }}

        .metric-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }}

        .metric-title {{
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
        }}

        .metric-value {{
            font-size: 1.85rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 0.25rem;
        }}

        .metric-subtext {{
            font-size: 0.8rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }}

        .metric-threshold {{
            font-size: 0.75rem;
            color: #94a3b8;
            border-top: 1px dashed var(--border);
            padding-top: 0.5rem;
        }}

        /* Tables */
        .table-wrap {{
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 10px;
            background-color: var(--surface);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.9rem;
        }}

        th {{
            background-color: #1a2234;
            color: #9ca3af;
            font-weight: 600;
            padding: 0.75rem 1rem;
            border-bottom: 1px solid var(--border);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}

        td {{
            padding: 0.75rem 1rem;
            border-bottom: 1px solid #1f2937;
            color: #e5e7eb;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover td {{
            background-color: #1e293b;
        }}

        code {{
            background-color: #1f2937;
            padding: 0.15rem 0.35rem;
            border-radius: 4px;
            font-size: 0.82rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            color: #38bdf8;
        }}

        .muted-text {{
            color: var(--text-muted);
        }}

        /* Dataset Stats Bar */
        .stats-bar {{
            display: flex;
            gap: 2rem;
            margin-bottom: 1rem;
            background-color: var(--surface);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            border: 1px solid var(--border);
            font-size: 0.95rem;
        }}

        .stats-bar div strong {{
            color: #ffffff;
            margin-left: 0.25rem;
        }}

        /* Audit Trail Accordions */
        .audit-card {{
            background-color: var(--surface);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin-bottom: 0.85rem;
            overflow: hidden;
        }}

        .audit-summary {{
            padding: 1rem 1.25rem;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 600;
            background-color: #161f30;
            user-select: none;
            transition: background 0.15s ease;
        }}

        .audit-summary:hover {{
            background-color: #1d283c;
        }}

        .audit-title {{
            font-size: 1rem;
            color: #f3f4f6;
        }}

        .audit-meta {{
            font-size: 0.82rem;
            color: var(--text-muted);
        }}

        .audit-body {{
            padding: 1.25rem;
            border-top: 1px solid var(--border);
            background-color: var(--surface);
        }}

        .audit-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }}

        @media (max-width: 768px) {{
            .audit-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .audit-col h4 {{
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--accent-cyan);
            margin-bottom: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}

        .kv-list {{
            list-style: none;
            font-size: 0.88rem;
        }}

        .kv-list li {{
            margin-bottom: 0.4rem;
            color: #d1d5db;
        }}

        .kv-list strong {{
            color: #9ca3af;
        }}

        pre {{
            background-color: #0b0f19;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 0.75rem;
            font-size: 0.8rem;
            overflow-x: auto;
            color: #a5f3fc;
        }}

        footer {{
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-top: 3rem;
            padding-top: 1.5rem;
            border-top: 1px solid var(--border);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="header-top">
                <h1>SyntheLoop Quality &amp; Optimization Report</h1>
                {status_badge}
            </div>
            <div class="header-meta">
                <span>Run ID: <strong>{clean_run_id}</strong></span>
                <span>Generated: <strong>{current_time}</strong></span>
                <span>Total Iterations: <strong>{total_iterations}</strong></span>
            </div>
        </header>

        <!-- Final Scorecard -->
        <section>
            <h2>Executive Quality Scorecard</h2>
            <div class="scorecard-grid">
                {ks_card}
                {corr_card}
                {js_card}
                {dcr_card}
                {util_card}
            </div>
        </section>

        <!-- Iteration Trends -->
        <section>
            <h2>Metric Trends Across Iterations</h2>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Iteration</th>
                            <th>KS Mean</th>
                            <th>Corr Diff</th>
                            <th>JS Div</th>
                            <th>Privacy DCR (5th%)</th>
                            <th>AUC Drop</th>
                            <th>Status</th>
                            <th>Action Taken</th>
                        </tr>
                    </thead>
                    <tbody>
                        {trend_table_html}
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Dataset Overview -->
        <section>
            <h2>Dataset Overview</h2>
            <div class="stats-bar">
                <div>Rows: <strong>{n_rows}</strong></div>
                <div>Columns: <strong>{n_cols}</strong></div>
                <div>Target Column: <strong>{html.escape(str(target_col))}</strong></div>
            </div>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Column</th>
                            <th>Dtype</th>
                            <th>Type</th>
                            <th>Missing</th>
                            <th>Unique</th>
                            <th>Summary / Top Distribution</th>
                        </tr>
                    </thead>
                    <tbody>
                        {cols_table_html}
                    </tbody>
                </table>
            </div>
        </section>

        <!-- Audit Trail -->
        <section>
            <h2>Full Audit Trail &amp; LLM Refinement History</h2>
            {audit_trail_html}
        </section>

        <footer>
            Generated automatically by <strong>SyntheLoop</strong> &bull; Closed-Loop Synthetic Data Generation
        </footer>
    </div>
</body>
</html>
"""


def save_report(html_content: str, run_id: str, output_dir: str = "outputs") -> Path:
    """Writes the HTML report string to {output_dir}/{run_id}/report.html.

    Args:
        html_content: The rendered HTML report string.
        run_id: Unique identifier for the pipeline run.
        output_dir: Base directory for output runs. Defaults to 'outputs'.

    Returns:
        Path: Path to the written report.html file.
    """
    run_dir = Path(output_dir) / str(run_id).strip()
    run_dir.mkdir(parents=True, exist_ok=True)
    report_file = run_dir / "report.html"

    report_file.write_text(html_content, encoding="utf-8")
    logger.info(f"Report saved to {report_file}")
    return report_file
