"""Test the test. We build fake seasons on top of the real schedule of situations, where
WE decide whether momentum exists, and check that the play-to-play test (P1)
  - stays quiet when there is no momentum, even when teams have good and bad halves, and
  - finds momentum, at the right size, when we plant it.
Same idea as the explainer's "Test the tests", on the real situations.
"""
import numpy as np
import pandas as pd
from perm import group_codes, perm_test, lag_pairs


def simulate_outcomes(p, sd_half=0.06, delta=0.0, shift_sd=0.0, seed=0):
    """y_t ~ Bernoulli(p_hat_t + u_half + delta * (y_{t-1} - p_hat_{t-1}) [+ mid-half shift]).

    sd_half: spread of team-half "good day" levels (no momentum, just levels).
    delta: planted momentum, in probability points per unit of last play's surprise.
    shift_sd: a level shift halfway through each half (injury, adjustment): slow drift, not momentum.
    """
    rng = np.random.default_rng(seed)
    g = group_codes(p, ["game_id", "posteam", "half"])
    ph = p.p_hat.values
    u = rng.normal(0, sd_half, g.max() + 1)[g]
    pos = p.groupby(g).cumcount().values
    size = np.bincount(g)[g]
    if shift_sd:
        u = u + (pos >= size // 2) * rng.normal(0, shift_sd, g.max() + 1)[g]
    y = np.zeros(len(p), dtype=int)
    prev_surprise = np.zeros(len(p))
    for t in range(pos.max() + 1):
        idx = np.where(pos == t)[0]
        pr = ph[idx] + u[idx]
        if t > 0 and delta:
            pr = pr + delta * prev_surprise[idx - 1]
        y[idx] = rng.random(len(idx)) < np.clip(pr, 0.01, 0.99)
        prev_surprise[idx] = y[idx] - ph[idx]
    return y


def run(p, n_sims=40, n_perm=200, settings=None):
    if settings is None:
        settings = {
            "No momentum, teams differ by half": dict(delta=0.0),
            "No momentum, plus mid-half level shifts": dict(delta=0.0, shift_sd=0.06),
            "Planted momentum: +0.5 pts": dict(delta=0.005),
            "Planted momentum: +1 pt": dict(delta=0.01),
            "Planted momentum: +2 pts": dict(delta=0.02),
        }
    rows = []
    g_half = group_codes(p, ["game_id", "posteam", "half"])
    a, b = lag_pairs(g_half)
    for name, kw in settings.items():
        for s in range(n_sims):
            y = simulate_outcomes(p, seed=1000 + s, **kw)
            r = y - p.p_hat.values
            res = perm_test(g_half, y, r, n_perm=n_perm, seed=s)
            raw = y[b][y[a] == 1].mean() - y[b][y[a] == 0].mean()
            rows.append(dict(setting=name, sim=s, planted=kw.get("delta", 0.0), raw=raw,
                             excess=res["excess"], z=res["z"], p=res["p"]))
    return pd.DataFrame(rows)
