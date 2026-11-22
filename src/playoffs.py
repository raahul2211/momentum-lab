"""Exploratory (not pre-registered): do streaky teams make the playoffs or win Super Bowls?

The team momentum index here is rebuilt on REGULAR-SEASON plays only, so a deep playoff run
can't feed the index it is being compared with. Outcomes come from games.csv, seasons 1999-2025
(2026 is still in progress).

Questions
  A. Do playoff teams and champions have higher (streakier) index values than everyone else?
  B. Holding team quality fixed, does the index add anything to predicting playoffs?
  C. Among playoff teams, does the index predict playoff wins or a title?
  D. Does this year's index predict next year's playoffs?
  E. Pooled: is there more play-to-play carryover in playoff games than in the regular season?
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exploratory import unit_index
from perm import group_codes, lag_pairs, within_group_perm

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(21)


def outcomes():
    g = pd.read_csv(ROOT / "data/raw/nfl/games.csv")
    g = g[(g.season <= 2025) & g.home_score.notna()]
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        t = g[["season", "game_type", f"{side}_team", f"{side}_score", f"{opp}_score"]].copy()
        t.columns = ["season", "game_type", "team", "pf", "pa"]
        rows.append(t)
    t = pd.concat(rows)
    t["team"] = t.team.replace({"OAK": "LV", "SD": "LAC", "STL": "LA"})  # match play-by-play codes
    t["win"] = (t.pf > t.pa).astype(float) + 0.5 * (t.pf == t.pa)
    reg = t[t.game_type == "REG"].groupby(["team", "season"]).agg(reg_wpct=("win", "mean"),
                                                                  pt_diff=("pf", "sum"), pa=("pa", "sum"), games=("win", "size"))
    reg["pt_diff"] = (reg.pt_diff - reg.pa) / reg.games
    post = t[t.game_type != "REG"]
    po = post.groupby(["team", "season"]).agg(po_games=("win", "size"), po_wins=("win", "sum"))
    sb = post[post.game_type == "SB"].groupby(["team", "season"]).win.max().rename("sb_win")
    o = reg.join(po).join(sb).reset_index()
    o["playoffs"] = o.po_games.notna().astype(int)
    o["po_wins"] = o.po_wins.fillna(0)
    o["sb_app"] = o.sb_win.notna().astype(int)
    o["sb_win"] = o.sb_win.fillna(0).astype(int)
    o["tier"] = np.select([o.sb_win == 1, o.sb_app == 1, o.playoffs == 1], ["Champion", "SB loser", "Playoffs"], "Missed")
    return o.drop(columns=["pa", "games"])


def perm_diff(x, grp, n=20000):
    """Difference in means (grp==1 minus grp==0) with a label-shuffle p-value and bootstrap CI."""
    x, grp = np.asarray(x), np.asarray(grp).astype(bool)
    obs = x[grp].mean() - x[~grp].mean()
    null = np.array([(lambda s: x[s].mean() - x[~s].mean())(rng.permutation(grp)) for _ in range(n)])
    a, b = x[grp], x[~grp]
    boots = [rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean() for _ in range(4000)]
    return dict(diff=obs, lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                p=float((np.abs(null) >= abs(obs)).mean()), n1=int(grp.sum()), n0=int((~grp).sum()))


def logit_lr(y, X_base, x_new):
    """Likelihood-ratio check: does adding the index improve a logistic model of y?"""
    import statsmodels.api as sm
    m0 = sm.Logit(y, sm.add_constant(X_base)).fit(disp=0)
    m1 = sm.Logit(y, sm.add_constant(np.column_stack([X_base, x_new]))).fit(disp=0)
    coef, se = m1.params[-1], m1.bse[-1]
    lr = 2 * (m1.llf - m0.llf)
    from scipy.stats import chi2
    return dict(coef_per_sd=float(coef), lo=float(coef - 1.96 * se), hi=float(coef + 1.96 * se),
                odds_ratio_per_sd=float(np.exp(coef)), p=float(chi2.sf(lr, 1)))


def pooled_p1(p, n_perm=500):
    p = p.sort_values(["game_id", "posteam", "half", "seq"]).reset_index(drop=True)
    g = group_codes(p, ["game_id", "posteam", "half"])
    a, b = lag_pairs(g)
    y, r = p.success.values, p.r.values

    def stat(yy, rr):
        prev, cur = yy[a], rr[b]
        return cur[prev == 1].mean() - cur[prev == 0].mean()
    obs = stat(y, r)
    null = np.array([stat(y[i], r[i]) for i in (within_group_perm(g, rng) for _ in range(n_perm))])
    ex = (obs - null.mean()) * 100
    return dict(excess_pts=ex, lo=ex - 196 * null.std(ddof=1), hi=ex + 196 * null.std(ddof=1),
                raw_pts=float((p.success.values[b][y[a] == 1].mean() - p.success.values[b][y[a] == 0].mean()) * 100),
                plays=int(len(p)), games=int(p.game_id.nunique()))


def main():
    p = pd.read_parquet(ROOT / "data/clean/nfl_plays_model.parquet")
    reg = p[(p.season_type == "REG") & (p.season <= 2025)]
    idx = unit_index(reg, ["posteam", "season"], n_perm=300).rename(columns={"posteam": "team"})
    o = outcomes()
    d = idx.merge(o, on=["team", "season"], how="inner")
    d["excess_z"] = (d.excess - d.excess.mean()) / d.excess.std()
    out = dict(n_team_seasons=len(d), seasons="1999-2025, regular-season plays only")

    # A. tiers
    out["tiers"] = (d.groupby("tier").agg(n=("excess", "size"), mean_excess=("excess", "mean"),
                                          median_excess=("excess", "median"), mean_z=("z", "mean"),
                                          share_streaky=("z", lambda z: float((z > 1.96).mean())),
                                          share_steady=("z", lambda z: float((z < -1.96).mean())))
                    .reindex(["Missed", "Playoffs", "SB loser", "Champion"]).reset_index().to_dict(orient="records"))
    out["playoff_vs_missed"] = perm_diff(d.excess, d.playoffs)
    out["champ_vs_rest"] = perm_diff(d.excess, d.sb_win)
    out["champ_vs_other_playoff"] = perm_diff(d[d.playoffs == 1].excess, d[d.playoffs == 1].sb_win)
    out["corr_index_wpct"] = dict(zip(["rho", "p"], map(float, spearmanr(d.excess, d.reg_wpct))))
    out["corr_level_wpct"] = dict(zip(["rho", "p"], map(float, spearmanr(d["level"], d.reg_wpct))))

    # B. holding quality fixed (play-level quality vs expectations + point differential)
    out["playoffs_given_quality"] = logit_lr(d.playoffs.values, d[["level", "pt_diff"]].values, d.excess_z.values)

    # C. among playoff teams
    pq = d[d.playoffs == 1]
    out["po_wins_corr"] = dict(zip(["rho", "p"], map(float, spearmanr(pq.excess, pq.po_wins))))
    out["sb_win_given_quality"] = logit_lr(pq.sb_win.values, pq[["level", "pt_diff"]].values, pq.excess_z.values)

    # D. next year
    d = d.sort_values(["team", "season"])
    d["next_playoffs"] = d.groupby("team").playoffs.shift(-1)
    d["next_season"] = d.groupby("team").season.shift(-1)
    nx = d[(d.next_season == d.season + 1)].dropna(subset=["next_playoffs"])
    out["next_year_playoffs"] = logit_lr(nx.next_playoffs.values, nx[["level", "pt_diff"]].values, nx.excess_z.values)
    out["next_year_n"] = len(nx)

    # E. pooled carryover, playoffs vs regular season
    post = p[(p.season_type == "POST") & (p.season <= 2025)]
    out["pooled_post"] = pooled_p1(post)
    out["pooled_reg"] = pooled_p1(reg, n_perm=200)

    # champions list for the page
    ch = d[d.sb_win == 1].sort_values("excess")[["team", "season", "excess", "z", "reg_wpct"]]
    out["champions"] = ch.to_dict(orient="records")
    out["playoff_rank_note"] = dict(n_playoff=int(d.playoffs.sum()))

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 5)
        if isinstance(o, np.integer):
            return int(o)
        return o
    (ROOT / "results/playoffs_nfl.json").write_text(json.dumps(clean(out), indent=1))
    d.to_csv(ROOT / "results/team_seasons_reg_index.csv", index=False)
    for k, v in out.items():
        if k != "champions":
            print(k, v)
    print(ch.to_string(index=False))


if __name__ == "__main__":
    main()
