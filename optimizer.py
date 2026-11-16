"""Mega 6/45 research engine.

Important: draws are assumed fair and independent unless robust out-of-sample
evidence rejects that assumption. The engine never presents historical hot
numbers as a guaranteed edge.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
import numpy as np
import pandas as pd

N_NUMBERS = 45
PICK = 6
N_COMBINATIONS = math.comb(N_NUMBERS, PICK)


def _draw_matrix(draws):
    """Return sorted integer draws as a (n, 6) numpy array."""
    rows = []
    for nums in draws:
        a = sorted({int(x) for x in nums})
        if len(a) != PICK or min(a) < 1 or max(a) > N_NUMBERS:
            continue
        rows.append(a)
    return np.asarray(rows, dtype=np.int16).reshape(-1, PICK)


def _marginal_probabilities(history, alpha=30.0, recent_window=100, recent_weight=0.0):
    """Dirichlet-shrunk number probabilities, with optional recency blend."""
    n = len(history)
    counts = np.bincount(history.ravel(), minlength=46)[1:].astype(float)
    # Each draw contributes six selections. Under uniformity, p(number)=6/45.
    base = np.full(45, PICK / N_NUMBERS)
    p_long = (counts + alpha * base) / (PICK * n + alpha * PICK)
    if recent_weight <= 0 or n == 0:
        return p_long
    recent = history[-min(recent_window, n):]
    rc = np.bincount(recent.ravel(), minlength=46)[1:].astype(float)
    p_recent = (rc + alpha * base) / (PICK * len(recent) + alpha * PICK)
    p = (1 - recent_weight) * p_long + recent_weight * p_recent
    return p / p.sum() * PICK


def _draw_log_score(draw, p):
    """Approximate log probability of a six-number set under weighted sampling
    without replacement (Plackett-Luce sequential approximation).
    """
    remaining = np.asarray(p, dtype=float).copy()
    score = 0.0
    for number in draw:
        idx = int(number) - 1
        denom = remaining.sum()
        score += math.log(max(remaining[idx] / denom, 1e-15))
        remaining[idx] = 0.0
    return score


def walk_forward_backtest(draws, warmup=100, recent_window=100):
    """Strict chronological evaluation; each forecast sees prior draws only.

    Models:
      uniform: fixed fair-draw baseline
      shrunk_frequency: long-run frequency with Dirichlet shrinkage
      recency_blend: shrunk long-run + recent frequency
    Returns per-draw scores and aggregate diagnostics.
    """
    data = _draw_matrix(draws)
    if len(data) <= warmup:
        raise ValueError(f"Need more than {warmup} valid draws; got {len(data)}.")
    records = []
    uniform = np.full(45, PICK / N_NUMBERS)
    for t in range(warmup, len(data)):
        hist, actual = data[:t], data[t]
        p_freq = _marginal_probabilities(hist, alpha=60.0)
        p_recent = _marginal_probabilities(
            hist, alpha=60.0, recent_window=recent_window, recent_weight=0.25
        )
        scores = {
            "uniform": _draw_log_score(actual, uniform),
            "shrunk_frequency": _draw_log_score(actual, p_freq),
            "recency_blend": _draw_log_score(actual, p_recent),
        }
        row = {"draw_index": t, **scores}
        for name, p in (("shrunk_frequency", p_freq), ("recency_blend", p_recent)):
            row[f"{name}_brier"] = float(np.mean(
                (np.bincount(actual, minlength=46)[1:] - p) ** 2
            ))
        records.append(row)
    result = pd.DataFrame(records)
    summary = []
    for model in ("shrunk_frequency", "recency_blend"):
        diff = result[model] - result["uniform"]
        # Positive log-score difference is better. Paired standard error.
        se = diff.std(ddof=1) / math.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        summary.append({
            "model": model,
            "mean_log_score": result[model].mean(),
            "uniform_mean_log_score": result["uniform"].mean(),
            "delta_vs_uniform": diff.mean(),
            "delta_se": se,
            "z_vs_uniform": diff.mean() / se if se and se > 0 else 0.0,
            "win_rate": float((diff > 0).mean()),
            "draws_tested": len(result),
        })
    return result, pd.DataFrame(summary)


def _portfolio_score(ticket, p):
    # Sum of marginal log probabilities. This ranks candidates only;
    # it is not a calibrated jackpot probability.
    return sum(math.log(max(p[n - 1], 1e-15)) for n in ticket)


def generate_portfolio(draws, budget=50000, price_per_line=10000,
                       strategy="diversified", seed=645):
    """Generate unique lines. Fair-draw mode uses reproducible random lines.
    Experimental mode ranks a bounded candidate pool using shrunk frequencies.
    """
    n_lines = int(budget // price_per_line)
    if n_lines < 1:
        raise ValueError("Budget is below the price of one line.")
    data = _draw_matrix(draws)
    rng = np.random.default_rng(seed)
    if strategy == "frequency_experimental" and len(data) >= 100:
        p = _marginal_probabilities(data, alpha=60.0, recent_window=100,
                                    recent_weight=0.20)
        # Generate a large candidate pool using weighted sampling without replacement.
        candidates = set()
        target = max(20000, n_lines * 5000)
        while len(candidates) < target:
            candidates.add(tuple(sorted(rng.choice(
                np.arange(1, 46), size=6, replace=False, p=p / p.sum()
            ).tolist())))
        ranked = sorted(candidates, key=lambda x: _portfolio_score(x, p), reverse=True)
        chosen = []
        used_numbers = set()
        for ticket in ranked:
            # Soft diversification: prefer new numbers while retaining model score.
            overlap = max((len(set(ticket) & set(x)) for x in chosen), default=0)
            adjusted = _portfolio_score(ticket, p) - 0.12 * overlap
            if not chosen or adjusted >= _portfolio_score(chosen[-1], p) - 1.5:
                chosen.append(ticket)
            if len(chosen) >= n_lines:
                break
        while len(chosen) < n_lines:
            ticket = tuple(sorted(rng.choice(np.arange(1, 46), 6, replace=False).tolist()))
            if ticket not in chosen:
                chosen.append(ticket)
    else:
        chosen = []
        seen = set()
        while len(chosen) < n_lines:
            ticket = tuple(sorted(rng.choice(np.arange(1, 46), 6, replace=False).tolist()))
            if ticket not in seen:
                seen.add(ticket)
                chosen.append(ticket)
    return pd.DataFrame([
        {"Vé": i + 1, "Số": " – ".join(f"{n:02d}" for n in ticket),
         "Numbers": ticket}
        for i, ticket in enumerate(chosen)
    ])


def theoretical_jackpot_probability(n_unique_lines):
    n = min(max(int(n_unique_lines), 0), N_COMBINATIONS)
    return n / N_COMBINATIONS


def portfolio_coverage(portfolio):
    """Descriptive coverage: distinct numbers and pairs covered by the tickets."""
    tickets = [tuple(row) for row in portfolio["Numbers"]]
    number_coverage = len(set(n for t in tickets for n in t))
    pair_coverage = len(set(pair for t in tickets for pair in combinations(t, 2)))
    return {
        "unique_lines": len(set(tickets)),
        "numbers_covered": number_coverage,
        "pairs_covered": pair_coverage,
        "max_pair_coverage": len(tickets) * math.comb(6, 2),
    }
