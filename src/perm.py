"""Permutation engine: "each game's own coin", generalized.

The Side Quest explainer's fair test shuffles each game's own possessions. We do the
same thing, with two upgrades:

1. We shuffle situation-adjusted residuals (actual minus expected), not raw results,
   so down/distance/field/score/clock and pregame strength are out before we look
   for order. (Difficulty adjustment: Bocskocsky, Ezekowitz & Stein 2014.)
2. We shuffle inside small blocks (a team's plays in one half), which keeps that
   team's level that half ("better that day") in the null. Any surviving signal is
   about ORDER, which is what momentum claims.

Shuffling also fixes the streak-selection bias (Miller & Sanjurjo 2018): the null is
computed on the same finite sequences, so the bias is in both observed and null.
For regressions with covariates we use Freedman & Lane (1983).
"""
import numpy as np


def group_codes(df, cols):
    return df.groupby(cols, sort=False).ngroup().values


def within_group_perm(g, rng):
    """Permutation index that shuffles rows only within contiguous groups g (sorted)."""
    return np.argsort(g + rng.random(len(g)), kind="stable")


def lag_pairs(g, lag=1):
    i = np.arange(lag, len(g))
    ok = g[i] == g[i - lag]
    return i[ok] - lag, i[ok]


def diff_stat(prev_y, cur_r):
    """Mean residual after a success minus after a failure (probability points)."""
    s = prev_y == 1
    return cur_r[s].mean() - cur_r[~s].mean()


def corr_stat(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())


def perm_test(g, y, r, n_perm=1000, lag=1, pair_mask=None, stat="diff", seed=1):
    """Order test. g: contiguous group codes; y: binary prior outcome; r: residual.

    Returns dict with observed, null mean, null sd, excess (obs - null mean), p (two-sided).
    pair_mask: optional boolean over the pair list (positions are fixed; values shuffle).
    """
    rng = np.random.default_rng(seed)
    a, b = lag_pairs(g, lag)
    if pair_mask is not None:
        a, b = a[pair_mask], b[pair_mask]
    f = (lambda yy, rr: diff_stat(yy[a], rr[b])) if stat == "diff" else (lambda yy, rr: corr_stat(rr[a], rr[b]))
    obs = f(y, r)
    null = np.empty(n_perm)
    for k in range(n_perm):
        idx = within_group_perm(g, rng)
        null[k] = f(y[idx], r[idx])
    mu, sd = null.mean(), null.std(ddof=1)
    p = (1 + np.sum(np.abs(null - mu) >= abs(obs - mu))) / (n_perm + 1)
    return dict(obs=obs, null_mean=mu, null_sd=sd, excess=obs - mu, z=(obs - mu) / sd, p=p,
                n_pairs=len(a), mde80=2.8 * sd, null=null)


# ---------- Freedman-Lane for y ~ x + Z + group FE ----------

def demean(v, g):
    v = np.asarray(v, float)
    if v.ndim == 1:
        v = v[:, None]
    sums = np.zeros((g.max() + 1, v.shape[1]))
    np.add.at(sums, g, v)
    cnt = np.bincount(g, minlength=g.max() + 1)[:, None]
    return v - (sums / np.maximum(cnt, 1))[g]


def ols(Y, X):
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta


def freedman_lane(g, y, x, Z=None, n_perm=1000, seed=2):
    """Coefficient on x in y ~ x + Z + FE(g), with Freedman-Lane permutation p-value.

    Permutes residuals of the reduced model (y ~ Z + FE) within groups.
    Groups must be contiguous (sorted).
    """
    rng = np.random.default_rng(seed)
    yd = demean(y, g)[:, 0]
    xd = demean(x, g)
    Zd = demean(Z, g) if Z is not None else np.zeros((len(y), 0))
    Xfull = np.column_stack([xd, Zd])
    obs = ols(yd, Xfull)[0]
    if Zd.shape[1]:
        gam = ols(yd, Zd)
        fit = Zd @ gam
    else:
        fit = np.zeros_like(yd)
    e = yd - fit
    null = np.empty(n_perm)
    for k in range(n_perm):
        idx = within_group_perm(g, rng)
        ystar = fit + e[idx]
        null[k] = ols(ystar, Xfull)[0]
    mu, sd = null.mean(), null.std(ddof=1)
    p = (1 + np.sum(np.abs(null - mu) >= abs(obs - mu))) / (n_perm + 1)
    return dict(obs=obs, null_mean=mu, null_sd=sd, excess=obs - mu, z=(obs - mu) / sd, p=p,
                n=len(y), mde80=2.8 * sd, null=null)


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (dict name->p)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out
