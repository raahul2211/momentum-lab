"""All hypothesis tests. Every function takes a season filter so the same code runs on
the discovery seasons and, once, on the holdout seasons named in PREREG.md.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from perm import group_codes, lag_pairs, within_group_perm, diff_stat, perm_test, freedman_lane

N_PERM = 2000


def _summ(res, scale=1.0):
    return {k: (v * scale if k in ("obs", "null_mean", "null_sd", "excess", "mde80") else v)
            for k, v in res.items() if k != "null"}


# ------------------------------------------------------------------ plays

def prep_plays(p, seasons):
    p = p[p.season.isin(seasons)].sort_values(["game_id", "posteam", "half", "seq"]).reset_index(drop=True)
    return p


def p1_play_carryover(p, n_perm=N_PERM, block=("game_id", "posteam", "half")):
    """P1: after a successful play, is the next play (same offense, same half) better than
    expected, beyond what a shuffle of that offense's plays in that half gives?
    Statistic in probability points (x100 = percentage points)."""
    g = group_codes(p, list(block))
    out = perm_test(g, p.success.values, p.r.values, n_perm=n_perm)
    return out


def play_waterfall(p, n_perm=N_PERM):
    """Where does raw 'success follows success' go? raw -> situation -> team-half -> order."""
    g = group_codes(p, ["game_id", "posteam", "half"])
    y, r = p.success.values, p.r.values
    a, b = lag_pairs(g)
    raw = y[b][y[a] == 1].mean() - y[b][y[a] == 0].mean()
    adj = diff_stat(y[a], r[b])
    res = perm_test(g, y, r, n_perm=n_perm)
    return dict(raw=raw, after_situation=adj, team_half_level=res["null_mean"], order=res["excess"],
                order_sd=res["null_sd"], order_p=res["p"], n_pairs=len(a),
                p_after_success=y[b][y[a] == 1].mean(), p_after_fail=y[b][y[a] == 0].mean())


def multi_mask_perm(g, y, r, masks, n_perm=N_PERM, lag=1, seed=11):
    """Same permutations, several pair subsets -> joint null for contrasts."""
    rng = np.random.default_rng(seed)
    a, b = lag_pairs(g, lag)
    obs = {k: diff_stat(y[a][m], r[b][m]) for k, m in masks.items()}
    null = {k: np.empty(n_perm) for k in masks}
    for i in range(n_perm):
        idx = within_group_perm(g, rng)
        yy, rr = y[idx], r[idx]
        for k, m in masks.items():
            null[k][i] = diff_stat(yy[a][m], rr[b][m])
    return obs, null, (a, b)


def contrast(obs, null, wts):
    o = sum(w * obs[k] for k, w in wts.items())
    n = sum(w * null[k] for k, w in wts.items())
    mu, sd = n.mean(), n.std(ddof=1)
    p = (1 + np.sum(np.abs(n - mu) >= abs(o - mu))) / (len(n) + 1)
    return dict(obs=o, null_mean=mu, excess=o - mu, null_sd=sd, z=(o - mu) / sd, p=p)


def excess_of(obs, null, k):
    return contrast(obs, null, {k: 1.0})


def decay_profile(p, lags=range(1, 9), n_perm=500):
    g = group_codes(p, ["game_id", "posteam", "half"])
    rows = []
    for L in lags:
        res = perm_test(g, p.success.values, p.r.values, n_perm=n_perm, lag=L, seed=100 + L)
        rows.append(dict(lag=L, **_summ(res)))
    return pd.DataFrame(rows)


def null_granularity(p, n_perm=1000):
    """How the answer depends on what we hold fixed ("each game's own coin" vs smaller blocks)."""
    out = []
    for name, block in [("whole game", ["game_id", "posteam"]), ("half", ["game_id", "posteam", "half"]),
                        ("quarter", ["game_id", "posteam", "qtr"]), ("drive", ["game_id", "posteam", "fixed_drive"])]:
        q = p.sort_values(["game_id", "posteam"] + [c for c in block if c not in ("game_id", "posteam")] + ["seq"]).reset_index(drop=True)
        g = group_codes(q, block)
        res = perm_test(g, q.success.values, q.r.values, n_perm=n_perm, seed=7)
        out.append(dict(block=name, **_summ(res)))
    return pd.DataFrame(out)


def leverage_and_crowd(p, n_perm=N_PERM):
    """Secondary: does carryover grow with the stakes of the last play (psychological 'mass',
    Adler 1981; Markman & Guenther 2007), and is it bigger for the home offense (crowd)?"""
    g = group_codes(p, ["game_id", "posteam", "half"])
    y, r = p.success.values, p.r.values
    a, b = lag_pairs(g)
    lev = p.lev_pre.values
    q80 = np.nanquantile(lev, 0.8)
    q20 = np.nanquantile(lev, 0.2)
    home = p.home.values[b] == 1
    masks = {
        "high_stakes": lev[a] >= q80, "low_stakes": lev[a] <= q20,
        "home_off": home, "away_off": ~home,
    }
    obs, null, _ = multi_mask_perm(g, y, r, masks, n_perm=n_perm)
    return dict(
        high_stakes=excess_of(obs, null, "high_stakes"), low_stakes=excess_of(obs, null, "low_stakes"),
        stakes_gap=contrast(obs, null, {"high_stakes": 1, "low_stakes": -1}),
        home=excess_of(obs, null, "home_off"), away=excess_of(obs, null, "away_off"),
        home_minus_away=contrast(obs, null, {"home_off": 1, "away_off": -1}),
    )


def crowd_2020(p, n_perm=N_PERM):
    """(home - away carryover) in 2020 minus the same in 2019 and 2021."""
    g = group_codes(p, ["game_id", "posteam", "half"])
    y, r = p.success.values, p.r.values
    a, b = lag_pairs(g)
    home = p.home.values[b] == 1
    s = p.season.values[b]
    masks = {"h20": home & (s == 2020), "a20": ~home & (s == 2020),
             "hX": home & np.isin(s, [2019, 2021]), "aX": ~home & np.isin(s, [2019, 2021])}
    obs, null, _ = multi_mask_perm(g, y, r, masks, n_perm=n_perm)
    return dict(did=contrast(obs, null, {"h20": 1, "a20": -1, "hX": -1, "aX": 1}),
                gap_2020=contrast(obs, null, {"h20": 1, "a20": -1}),
                gap_other=contrast(obs, null, {"hX": 1, "aX": -1}))


def big_moments(p, n_perm=N_PERM):
    """Secondary: next play after a 4th-down conversion vs after a 3rd-down conversion.
    Both hand the offense a fresh set of downs; only one is a big emotional moment."""
    g = group_codes(p, ["game_id", "posteam", "half"])
    a, b = lag_pairs(g)
    same_drive = p.fixed_drive.values[a] == p.fixed_drive.values[b]
    prev_down = p.down.values[a]
    conv = (p.success.values[a] == 1) & same_drive & np.isin(prev_down, [3, 4]) & (p.down.values[b] == 1)
    a, b = a[conv], b[conv]
    fourth = (prev_down[conv] == 4).astype(float)
    r = p.r.values[b]
    gg = g[b]
    order = np.argsort(gg, kind="stable")
    gg2 = pd.factorize(gg[order])[0]
    res = freedman_lane(gg2, r[order], fourth[order], None, n_perm=n_perm)
    return dict(res, n_fourth=int(fourth.sum()), n_third=int((1 - fourth).sum()),
                mean_r_after_4th=r[fourth == 1].mean(), mean_r_after_3rd=r[fourth == 0].mean())


# ------------------------------------------------------------------ drives

def drive_pairs(d, seasons):
    d = d[d.season.isin(seasons)].sort_values(["game_id", "fixed_drive"]).reset_index(drop=True)
    d["nxt_team"] = d.groupby("game_id").posteam.shift(-1)
    d["nxt_half"] = d.groupby("game_id").half.shift(-1)
    for c in ["poe", "sr_sum", "epa_oe", "duration", "n_plays"]:
        d["nxt_" + c] = d.groupby("game_id")[c].shift(-1)
    return d


def seq_shuffle_partial(g, x, Z, n_perm=N_PERM, seed=21):
    """Slope of x[k+1] on x[k] controlling Z[k], pooled over contiguous groups g.
    Null: shuffle each group's rows (x and Z travel together) and recompute. This keeps
    the exact finite-sequence (Nickell / streak-selection) bias in the null, which a
    fixed-effects regression with residual permutation would not."""
    rng = np.random.default_rng(seed)
    a, b = lag_pairs(g)

    def coef(xx, ZZ):
        X = np.column_stack([np.ones(len(a)), xx[a], ZZ[a]])
        return np.linalg.lstsq(X, xx[b], rcond=None)[0][1]
    obs = coef(x, Z)
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = within_group_perm(g, rng)
        null[i] = coef(x[idx], Z[idx])
    mu, sd = null.mean(), null.std(ddof=1)
    p = (1 + np.sum(np.abs(null - mu) >= abs(obs - mu))) / (n_perm + 1)
    return dict(obs=obs, null_mean=mu, null_sd=sd, excess=obs - mu, z=(obs - mu) / sd, p=p, n=len(a),
                mde80=2.8 * sd, null=null)


def p2_same_unit(d, outcome="poe", n_perm=N_PERM):
    """P2: offense drive k -> same offense's next drive in the same half. Controls for
    drive k's length (seconds, plays): a long drive tires the other defense."""
    t = d.dropna(subset=[outcome, "duration"]).sort_values(["game_id", "posteam", "half", "fixed_drive"]).reset_index(drop=True)
    g = group_codes(t, ["game_id", "posteam", "half"])
    return seq_shuffle_partial(g, t[outcome].values, t[["duration", "n_plays"]].values.astype(float), n_perm=n_perm)


def p3_cross_unit(d, outcome="poe", n_perm=N_PERM):
    """P3: offense drive k -> OPPONENT's next drive (i.e., our defense), same half,
    controlling for how long our drive was (defensive rest). Different players, so a
    matchup the offense found cannot explain it. Momentum predicts a NEGATIVE slope."""
    t = d[(d.nxt_team == d.defteam) & (d.nxt_half == d.half)].copy()
    t = t.dropna(subset=[outcome, "nxt_" + outcome, "duration"])
    t["g"] = t.game_id.astype(str) + t.posteam.astype(str) + t.half.astype(str)
    t = t.sort_values(["g", "fixed_drive"]).reset_index(drop=True)
    g = pd.factorize(t.g)[0]
    res = freedman_lane(g, t["nxt_" + outcome].values, t[outcome].values, t[["duration", "n_plays"]].values,
                        n_perm=n_perm)
    return res


def breaks_test(d, outcome="poe", n_perm=N_PERM):
    """Secondary: same-offense drive carryover when the two drives are separated by
    nothing, by a quarter break (Q1->Q2, Q3->Q4), or by halftime."""
    t = d.dropna(subset=[outcome]).sort_values(["game_id", "posteam", "fixed_drive"]).reset_index(drop=True)
    g = group_codes(t, ["game_id", "posteam"])
    a, b = lag_pairs(g)
    qa, qb = t.end_qtr.values[a], t.qtr.values[b]
    halftime = (qa <= 2) & (qb >= 3)
    qbreak = ((qa == 1) & (qb == 2)) | ((qa == 3) & (qb == 4))
    none = qa == qb
    x = t[outcome].values
    rng = np.random.default_rng(5)

    def slope(xa, xb):
        xa = xa - xa.mean(); xb = xb - xb.mean()
        return (xa * xb).sum() / (xa * xa).sum()
    masks = dict(none=none, quarter_break=qbreak, halftime=halftime)
    obs = {k: slope(x[a][m], x[b][m]) for k, m in masks.items()}
    null = {k: np.empty(n_perm) for k in masks}
    for i in range(n_perm):
        xx = x[within_group_perm(g, rng)]
        for k, m in masks.items():
            null[k][i] = slope(xx[a][m], xx[b][m])
    out = {k: dict(excess_of(obs, null, k), n=int(m.sum())) for k, m in masks.items()}
    out["none_minus_halftime"] = contrast(obs, null, {"none": 1, "halftime": -1})
    out["none_minus_quarter"] = contrast(obs, null, {"none": 1, "quarter_break": -1})
    return out


# ------------------------------------------------------------------ fumbles

def fumble_table(f, d, p, seasons):
    """For each live-ball scrimmage fumble: who got the bounce, and how did each team's
    next drive go versus expectation? Same rule in both arms: the first drive by each team
    that starts after the fumble's drive, in the same half."""
    f = f[f.season.isin(seasons)].copy()
    d = d[d.season.isin(seasons)]
    dk = d.set_index(["game_id", "fixed_drive"]).sort_index()
    by_game = {gid: grp for gid, grp in d.groupby("game_id")}
    # pre-fumble in-game form: mean residual of each team's offense earlier in the game
    p = p[p.season.isin(seasons)]
    pg = {gid: grp for gid, grp in p.groupby("game_id")}
    rows = []
    for row in f.itertuples(index=False):
        gd = by_game.get(row.game_id)
        if gd is None:
            continue
        later = gd[(gd.fixed_drive > row.fixed_drive) & (gd.half == row.half)]
        a_next = later[later.posteam == row.posteam].head(1)
        b_next = later[later.posteam == row.defteam].head(1)
        gp = pg.get(row.game_id)
        before = gp[gp.fixed_drive < row.fixed_drive] if gp is not None else None
        a_form = before[before.posteam == row.posteam].r.mean() if before is not None else np.nan
        b_form = before[before.posteam == row.defteam].r.mean() if before is not None else np.nan
        after = gp[gp.fixed_drive > row.fixed_drive] if gp is not None else None
        rows.append(dict(
            game_id=row.game_id, season=row.season, posteam=row.posteam, defteam=row.defteam, lost=row.lost,
            ftype=row.ftype, forced=row.fumble_forced, qtr=row.qtr, yardline_100=row.yardline_100,
            half_secs=row.half_seconds_remaining, score_diff=row.score_differential, wp=row.wp,
            pos_spread=row.pos_spread, home=row.home, return_td=row.return_touchdown,
            a_form=a_form, b_form=b_form,
            a_next_poe=a_next.poe.iloc[0] if len(a_next) else np.nan,
            b_next_poe=b_next.poe.iloc[0] if len(b_next) else np.nan,
            a_next_sr=a_next.sr_sum.iloc[0] if len(a_next) else np.nan,
            b_next_sr=b_next.sr_sum.iloc[0] if len(b_next) else np.nan,
            a_next_epa=a_next.epa_oe.iloc[0] if len(a_next) else np.nan,
            b_next_epa=b_next.epa_oe.iloc[0] if len(b_next) else np.nan,
            a_rest_r=after[after.posteam == row.posteam].r.mean() if after is not None else np.nan,
            b_rest_r=after[after.posteam == row.defteam].r.mean() if after is not None else np.nan,
        ))
    t = pd.DataFrame(rows)
    t["swing"] = t.b_next_poe - t.a_next_poe
    t["swing_sr"] = t.b_next_sr - t.a_next_sr
    t["swing_epa"] = t.b_next_epa - t.a_next_epa
    t["rest_swing"] = t.b_rest_r - t.a_rest_r
    return t


def fumble_balance(t):
    """Is the bounce as good as random? Pre-fumble covariates by arm, standardized
    differences, and a logit of 'lost' on pre-fumble form within fumble type."""
    cov = ["yardline_100", "half_secs", "score_diff", "wp", "pos_spread", "home", "a_form", "b_form", "forced"]
    rows = []
    for c in cov:
        x1, x0 = t.loc[t.lost == 1, c].dropna(), t.loc[t.lost == 0, c].dropna()
        sd = np.sqrt((x1.var() + x0.var()) / 2)
        rows.append(dict(covariate=c, lost_mean=x1.mean(), kept_mean=x0.mean(), std_diff=(x1.mean() - x0.mean()) / sd))
    bal = pd.DataFrame(rows)
    forced = " + forced" if t.forced.nunique() > 1 else ""
    m = smf.logit("lost ~ C(ftype) + a_form + b_form + score_diff + wp + pos_spread + home + yardline_100" + forced,
                  data=t.dropna(subset=["a_form", "b_form", "wp", "pos_spread", "forced"])).fit(disp=0)
    return bal, m


def p4_fumble(t, outcome="swing", n_perm=N_PERM, seed=4):
    """P4: effect of losing a live fumble (vs keeping it) on the next-possession swing
    (opponent's next-drive POE minus fumbling team's next-drive POE). Positive = momentum.
    Randomization inference: shuffle 'lost' within fumble type x forced strata."""
    s = t.dropna(subset=[outcome, "a_form", "b_form", "wp", "pos_spread", "forced"]).copy()
    s["stratum"] = s.ftype + s.forced.astype(int).astype(str)
    form = f"{outcome} ~ lost + C(stratum) + yardline_100 + half_secs + score_diff + wp + pos_spread + home + a_form + b_form"
    m = smf.ols(form, data=s).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(s.game_id)[0]})
    est = m.params["lost"]
    # randomization inference on the covariate-adjusted estimate
    rng = np.random.default_rng(seed)
    X = m.model.exog.copy()
    j = m.model.exog_names.index("lost")
    yv = m.model.endog
    strata = s.stratum.values
    null = np.empty(n_perm)
    groups = [np.where(strata == k)[0] for k in np.unique(strata)]
    for i in range(n_perm):
        Xp = X.copy()
        col = Xp[:, j].copy()
        for gi in groups:
            col[gi] = col[gi][rng.permutation(len(gi))]
        Xp[:, j] = col
        null[i] = np.linalg.lstsq(Xp, yv, rcond=None)[0][j]
    p_ri = (1 + np.sum(np.abs(null) >= abs(est))) / (n_perm + 1)
    raw = s.groupby("lost")[outcome].mean()
    return dict(est=est, se=m.bse["lost"], ci=tuple(m.conf_int().loc["lost"]), p_cluster=m.pvalues["lost"],
                p_ri=p_ri, n=len(s), n_lost=int(s.lost.sum()), raw_lost=raw.get(1), raw_kept=raw.get(0),
                mde80=2.8 * m.bse["lost"])
