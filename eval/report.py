"""
Human-readable reporting over ModelMetrics collections.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .metrics import CATEGORIES, ModelMetrics


def _fmt_pct(x: float) -> str:
    return f"{x * 100:5.1f}"


def format_model_line(m: ModelMetrics) -> str:
    """One-liner with macro-F1 and per-category F1."""
    cat_parts = []
    for cat in CATEGORIES:
        c = m.by_category.get(cat)
        cat_parts.append(f"{cat[:4]}={_fmt_pct(c.f1) if c else '  -  '}")
    macro = _fmt_pct(m.macro_f1_categories)
    return (
        f"{m.model:<40} macro_F1={macro}  "
        + "  ".join(cat_parts)
        + f"  n={m.n_examples}"
    )


def format_summary(models: List[ModelMetrics]) -> str:
    """Full multi-model report."""
    lines: List[str] = []
    lines.append("=" * 110)
    lines.append("Evaluation summary")
    lines.append("=" * 110)
    lines.append("")

    # --- Headline: macro-F1 ranking -----------------------------------------
    ranked = sorted(models, key=lambda m: m.macro_f1_categories, reverse=True)
    lines.append("Ranking by macro-F1 across categories:")
    lines.append("-" * 110)
    lines.append(
        f"{'model':<40} {'macro_F1':>9}  "
        + "  ".join(f"{cat[:8]:>8}" for cat in CATEGORIES)
        + f"  {'n':>3}"
    )
    for m in ranked:
        row = f"{m.model:<40} {_fmt_pct(m.macro_f1_categories):>9}  "
        for cat in CATEGORIES:
            c = m.by_category.get(cat)
            row += f"{(_fmt_pct(c.f1) if c else '   -   '):>8}  "
        row += f"{m.n_examples:>3}"
        lines.append(row)
    lines.append("")

    # --- Overall precision / recall / F1 ------------------------------------
    lines.append("Overall (micro-averaged across all fields):")
    lines.append("-" * 110)
    lines.append(
        f"{'model':<40} {'P':>6} {'R':>6} {'F1':>6} "
        f"{'TP':>5} {'FN':>5} {'FPw':>5} {'FPh':>5} {'TN':>5} "
        f"{'halluc%':>8} {'err':>4} {'t,s':>6}"
    )
    for m in ranked:
        o = m.overall
        lines.append(
            f"{m.model:<40} "
            f"{_fmt_pct(o.precision):>6} {_fmt_pct(o.recall):>6} {_fmt_pct(o.f1):>6} "
            f"{o.tp:>5} {o.fn:>5} {o.fp_wrong:>5} {o.fp_halluc:>5} {o.tn:>5} "
            f"{_fmt_pct(o.hallucination_rate):>8} "
            f"{m.errors:>4} {m.avg_latency_sec:>6.1f}"
        )
    lines.append("")

    # --- Per-category breakdown for each model ------------------------------
    lines.append("Per-category P / R / F1:")
    lines.append("-" * 110)
    for m in ranked:
        lines.append(f"  {m.model}")
        for cat in CATEGORIES:
            c = m.by_category.get(cat)
            if c is None or (c.tp + c.fn + c.fp_wrong + c.fp_halluc) == 0:
                lines.append(f"    {cat:<12} (no data)")
                continue
            lines.append(
                f"    {cat:<12} "
                f"P={_fmt_pct(c.precision)}  R={_fmt_pct(c.recall)}  F1={_fmt_pct(c.f1)}  "
                f"TP={c.tp:>3} FN={c.fn:>3} FPw={c.fp_wrong:>3} FPh={c.fp_halluc:>3} TN={c.tn:>3}"
            )
        lines.append("")

    return "\n".join(lines)


def model_metrics_to_dict(m: ModelMetrics) -> Dict[str, Any]:
    """JSON-serializable view of a ModelMetrics instance."""
    def dump(c):
        return {
            "tp": c.tp, "fn": c.fn,
            "fp_wrong": c.fp_wrong, "fp_halluc": c.fp_halluc, "tn": c.tn,
            "precision": c.precision, "recall": c.recall, "f1": c.f1,
            "support": c.support, "hallucination_rate": c.hallucination_rate,
        }

    return {
        "model": m.model,
        "backend": m.backend,
        "vllm_extra_args": m.vllm_extra_args,
        "n_examples": m.n_examples,
        "macro_f1_categories": m.macro_f1_categories,
        "avg_latency_sec": m.avg_latency_sec,
        "avg_attempts": m.avg_attempts,
        "errors": m.errors,
        "overall": dump(m.overall),
        "by_category": {k: dump(v) for k, v in m.by_category.items()},
        "by_field": {k: dump(v) for k, v in m.by_field.items()},
    }
