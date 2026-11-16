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
    """Exact log probability of an unordered six-number set under a
    Plackett-Luce sequential-without-replacement model.

    Dynamic programming sums the probabilities of all 6! possible draw
    orders. A DP state is the subset of target numbers already selected.
    """
    weights = np.asarray([p[int(number) - 1] for number in draw], dtype=float)
    total = float(np.sum(p))
    if np.any(weights <= 0) or total <= 0:
        return math.log(1e-300)

    # dp[mask] sums the probability of every ordering that selected mask.
    dp = np.zeros(1 << len(weights), dtype=float)
    dp[0] = 1.0
    subset_weight = np.zeros(1 << len(weights), dtype=float)
    for mask in range(1, len(subset_weight)):
        bit = mask & -mask
        idx = bit.bit_length() - 1
        subset_weight[mask] = subset_weight[mask ^ bit] + weights[idx]

    full = (1 << len(weights)) - 1
    for mask in range(full):
        denom = total - subset_weight[mask]
        if denom <= 0:
            continue
        remaining = full ^ mask
        while remaining:
            bit = remaining & -remaining
            idx = bit.bit_length() - 1
            dp[mask | bit] += dp[mask] * weights[idx] / denom
            remaining ^= bit

    return math.log(max(float(dp[full]), 1e-300))


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


def make_disjoint_portfolio(seed=645, n_lines=5):
    """Construct tickets with no shared numbers; maximizes unique-number,
    pair and triple coverage for n_lines <= 7 (subject to 45 numbers).
    """
    if n_lines * PICK > N_NUMBERS:
        raise ValueError("Cannot make all tickets number-disjoint with this many lines.")
    rng = np.random.default_rng(seed)
    numbers = rng.permutation(np.arange(1, 46))[:n_lines * PICK]
    tickets = [tuple(sorted(map(int, numbers[i*PICK:(i+1)*PICK])))
               for i in range(n_lines)]
    return pd.DataFrame([
        {"Vé": i + 1, "Số": " – ".join(f"{n:02d}" for n in ticket),
         "Numbers": ticket}
        for i, ticket in enumerate(tickets)
    ])


def monte_carlo_portfolio_comparison(trials=50000, n_lines=5, seed=645):
    """Compare random and number-disjoint portfolios under fair Mega 6/45.

    Simulates a fresh random draw for each portfolio. Reports coverage and
    probability of at least one ticket achieving >=k matches. Monte Carlo
    estimates for small probabilities (>=5 matches) have wide uncertainty.
    """
    rng = np.random.default_rng(seed)
    output = []
    for strategy in ("random", "disjoint"):
        distinct_total = pair_total = triple_total = 0
        hit_counts = {k: 0 for k in range(2, 7)}
        for _ in range(trials):
            tickets = []
            used = set()
            for _line in range(n_lines):
                pool = np.array([x for x in range(1, 46)
                                 if strategy != "disjoint" or x not in used])
                ticket = tuple(sorted(map(int, rng.choice(pool, PICK, replace=False))))
                tickets.append(ticket)
                if strategy == "disjoint":
                    used.update(ticket)
            distinct_total += len(set(x for t in tickets for x in t))
            pair_total += len(set(pair for t in tickets for pair in combinations(t, 2)))
            triple_total += len(set(triple for t in tickets for triple in combinations(t, 3)))
            draw = set(map(int, rng.choice(np.arange(1, 46), PICK, replace=False)))
            best = max(len(draw.intersection(t)) for t in tickets)
            for k in hit_counts:
                if best >= k:
                    hit_counts[k] += 1
        output.append({
            "strategy": strategy,
            "trials": trials,
            "avg_distinct_numbers": distinct_total / trials,
            "avg_unique_pairs": pair_total / trials,
            "avg_unique_triples": triple_total / trials,
            **{f"prob_at_least_{k}_hits": v / trials for k, v in hit_counts.items()},
            **{f"se_at_least_{k}_hits": math.sqrt((v / trials) * (1-v/trials) / trials)
               for k, v in hit_counts.items()},
        })
    return pd.DataFrame(output)


def bias_diagnostics(draws):
    """Multiplicity-adjusted diagnostics for marginal and pair frequencies,
    half-sample stability, and adjacent-draw overlap.
    """
    from scipy.stats import binomtest, hypergeom, chisquare

    data = _draw_matrix(draws)
    n = len(data)
    if n < 20:
        raise ValueError("At least 20 valid draws are required.")
    def adjust_bh(pvals, by=False):
        p = np.asarray(pvals, dtype=float)
        m = len(p)
        factor = sum(1/i for i in range(1, m+1)) if by else 1.0
        order = np.argsort(p)
        q = np.empty(m)
        running = 1.0
        for rank in range(m, 0, -1):
            idx = order[rank-1]
            running = min(running, p[idx] * m * factor / rank)
            q[idx] = min(1.0, running)
        return q

    number_counts = np.bincount(data.ravel(), minlength=46)[1:]
    number_p = [binomtest(int(c), n, PICK/N_NUMBERS).pvalue for c in number_counts]
    number_q = adjust_bh(number_p)
    number_by = adjust_bh(number_p, by=True)

    pair_counts = Counter()
    for draw in data:
        pair_counts.update(combinations(map(int, draw), 2))
    all_pairs = list(combinations(range(1, 46), 2))
    pair_p = [binomtest(pair_counts[pair], n, 1/66).pvalue for pair in all_pairs]
    pair_q = adjust_bh(pair_p)
    pair_by = adjust_bh(pair_p, by=True)

    half = n // 2
    c1 = np.bincount(data[:half].ravel(), minlength=46)[1:]
    c2 = np.bincount(data[half:].ravel(), minlength=46)[1:]
    pair1, pair2 = Counter(), Counter()
    for draw in data[:half]:
        pair1.update(combinations(map(int, draw), 2))
    for draw in data[half:]:
        pair2.update(combinations(map(int, draw), 2))
    pair_vec1 = np.array([pair1[p] for p in all_pairs], dtype=float)
    pair_vec2 = np.array([pair2[p] for p in all_pairs], dtype=float)

    overlap = np.zeros(PICK+1, dtype=int)
    for a, b in zip(data[:-1], data[1:]):
        overlap[len(set(map(int, a)).intersection(map(int, b)))] += 1
    overlap_n = n - 1
    overlap_prob = np.array([
        hypergeom.pmf(k, N_NUMBERS, PICK, PICK) for k in range(PICK+1)
    ])
    expected_overlap = overlap_prob * overlap_n
    # Combine tail categories with expected count < 5 for a valid chi-square approximation.
    obs = overlap.astype(float).copy()
    exp = expected_overlap.astype(float).copy()
    while len(exp) > 2 and exp[-1] < 5:
        obs[-2] += obs[-1]
        exp[-2] += exp[-1]
        obs, exp = obs[:-1], exp[:-1]
    overlap_chi, overlap_p = chisquare(obs, exp)

    number_rows = pd.DataFrame({
        "number": np.arange(1, 46), "count": number_counts,
        "p_value": number_p, "q_BH": number_q, "q_BY": number_by
    }).sort_values("count", ascending=False)
    pair_rows = pd.DataFrame({
        "pair": [f"{a:02d}-{b:02d}" for a, b in all_pairs],
        "count": [pair_counts[p] for p in all_pairs],
        "p_value": pair_p, "q_BH": pair_q, "q_BY": pair_by
    }).sort_values("count", ascending=False)
    return {
        "number_table": number_rows,
        "pair_table": pair_rows,
        "number_significant_BH_05": int((number_q < 0.05).sum()),
        "number_significant_BY_05": int((number_by < 0.05).sum()),
        "pair_significant_BH_05": int((pair_q < 0.05).sum()),
        "pair_significant_BY_05": int((pair_by < 0.05).sum()),
        "number_half_correlation": float(np.corrcoef(c1, c2)[0, 1]),
        "pair_half_correlation": float(np.corrcoef(pair_vec1, pair_vec2)[0, 1]),
        "adjacent_overlap_counts": overlap.tolist(),
        "adjacent_overlap_chi_square": float(overlap_chi),
        "adjacent_overlap_p_value": float(overlap_p),
        "expected_adjacent_overlap": overlap_prob.tolist(),
    }
