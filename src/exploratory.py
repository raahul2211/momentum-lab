"""Exploratory (not pre-registered): who plays 'with momentum'? Plus case studies.

1. Team-season and QB momentum index = P1 excess for that unit, with its own shuffle null.
   Funnel plot logic: if momentum were a team trait, more units would fall outside the
   chance bands than 5%, and the index would repeat year to year.
2. Broncos one-score games: how rare is 17-1, given each game's pregame odds?
3. Game timelines: Giants at Broncos 2025 (the video's opener) and Super Bowl XLVII.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perm import group_codes, lag_pairs, within_group_perm, diff_stat

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
RAW = ROOT / "data" / "raw" / "nfl"
OUT = ROOT / "results"


def unit_index(p, unit_cols, n_perm=300, min_pairs=300, seed=3):
    """P1 excess per unit (e.g., team-season). Shuffles within team-game-half as always."""
    p = p.sort_values(["game_id", "posteam", "half", "seq"]).reset_index(drop=True)
    g = group_codes(p, ["game_id", "posteam", "half"])
    a, b = lag_pairs(g)
    unit = p.groupby(unit_cols, sort=False).ngroup().values[b]
    y, r = p.success.values, p.r.values
    n_units = unit.max() + 1

    def per_unit(yy, rr):
        prev = yy[a]
        cur = rr[b]
        s1 = np.bincount(unit, weights=cur * (prev == 1), minlength=n_units)
        c1 = np.bincount(unit, weights=(prev == 1).astype(float), minlength=n_units)
        s0 = np.bincount(unit, weights=cur * (prev == 0), minlength=n_units)
        c0 = np.bincount(unit, weights=(prev == 0).astype(float), minlength=n_units)
        return s1 / np.maximum(c1, 1) - s0 / np.maximum(c0, 1)
    obs = per_unit(y, r)
    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, n_units))
    for i in range(n_perm):
        idx = within_group_perm(g, rng)
        null[i] = per_unit(y[idx], r[idx])
    mu, sd = null.mean(0), null.std(0, ddof=1)
    keys = p.iloc[b].groupby(unit, sort=True)[unit_cols].first().reset_index(drop=True)
    npairs = np.bincount(unit, minlength=n_units)
    out = keys.assign(obs=obs, null_mean=mu, null_sd=sd, excess=obs - mu, z=(obs - mu) / sd, n_pairs=npairs)
    # also the unit's overall level (how good vs expected), to show level != momentum
    lvl = p.groupby(unit_cols).agg(level=("r", "mean"), plays=("r", "size")).reset_index()
    out = out.merge(lvl, on=unit_cols)
    return out[out.n_pairs >= min_pairs].reset_index(drop=True)


def reliability(df, value="excess", sd="null_sd"):
    """Share of the spread across units that is NOT chance (empirical-Bayes reliability)."""
    v_obs = df[value].var()
    v_noise = (df[sd] ** 2).mean()
    return dict(var_obs=v_obs, var_chance=v_noise, reliability=max(0.0, 1 - v_noise / v_obs),
                share_outside_95=float((df.z.abs() > 1.96).mean()))


def team_seasons(p):
    t = unit_index(p, ["posteam", "season"])
    t = t.sort_values(["posteam", "season"])
    t["next_excess"] = t.groupby("posteam").excess.shift(-1)
    t["next_level"] = t.groupby("posteam")["level"].shift(-1)
    yoy = t.dropna(subset=["next_excess"])
    rel = reliability(t)
    rel["yoy_corr_momentum"] = float(np.corrcoef(yoy.excess, yoy.next_excess)[0, 1])
    rel["yoy_corr_level"] = float(np.corrcoef(yoy["level"], yoy["next_level"])[0, 1])
    # shrunk estimate toward 0 by reliability (the honest ranking)
    t["shrunk"] = t.excess * rel["reliability"]
    return t, rel


def qb_careers(p):
    """Primary passer of each team-game = most dropbacks; QB careers with >= 50 games."""
    q = p[p.passer_player_name.notna()].groupby(["game_id", "posteam", "passer_player_name"]).size().rename("n").reset_index()
    q = q.sort_values("n").groupby(["game_id", "posteam"]).tail(1)[["game_id", "posteam", "passer_player_name"]]
    pp = p.merge(q, on=["game_id", "posteam"], how="inner", suffixes=("_play", ""))
    games = pp.groupby("passer_player_name").game_id.nunique()
    keep = games[games >= 50].index
    pp = pp[pp.passer_player_name.isin(keep)]
    t = unit_index(pp, ["passer_player_name"], min_pairs=1000)
    t["games"] = t.passer_player_name.map(games)
    return t, reliability(t)


def spread_wp(spread_for_team, sd=13.45):
    """Pregame win probability from the point spread (normal margin, SD ~13.45; Stern 1991)."""
    return norm.cdf(spread_for_team / sd)


def broncos_close_games(n_sim=200000, seed=9):
    g = pd.read_csv(RAW / "games.csv")
    g = g[g.home_score.notna() & g.spread_line.notna()].copy()
    rows = []
    for side in ["home", "away"]:
        t = g.copy()
        t["team"] = t[f"{side}_team"]
        t["margin"] = np.where(side == "home", t.result, -t.result)
        t["spread_for"] = np.where(side == "home", t.spread_line, -t.spread_line)
        rows.append(t[["game_id", "season", "gameday", "game_type", "team", "margin", "spread_for"]])
    tg = pd.concat(rows).sort_values(["team", "gameday"]).reset_index(drop=True)
    tg = tg[tg.margin != 0]
    close = tg[tg.margin.abs() <= 8].copy()
    close["win"] = (close.margin > 0).astype(int)
    close["wp_pre"] = spread_wp(close.spread_for)
    # league history: best 18-game stretch of one-score games for every team
    best = []
    for team, x in close.groupby("team"):
        w = x.win.rolling(18).sum()
        if w.notna().any():
            i = int(w.idxmax())
            best.append(dict(team=team, best18=int(w.max()), end=x.loc[i, "gameday"], start=x.loc[x.index[x.index.get_loc(i) - 17], "gameday"]))
    best = pd.DataFrame(best).sort_values("best18", ascending=False)
    # Pregame odds overstate favorites once we condition on the game being close, so we
    # calibrate: logistic fit of close-game wins on the spread, using games before Oct 2025.
    import statsmodels.api as sm
    hist = close[close.gameday < "2025-10-01"]
    fit = sm.Logit(hist.win, sm.add_constant(hist.spread_for)).fit(disp=0)
    close["wp_close"] = fit.predict(sm.add_constant(close.spread_for))
    den = close[(close.team == "DEN") & (close.gameday >= "2025-10-01")].copy()
    rng = np.random.default_rng(seed)
    sims = (rng.random((n_sim, len(den))) < den.wp_close.values).sum(1)
    sims_pre = (rng.random((n_sim, len(den))) < den.wp_pre.values).sum(1)
    # How often does SOME team, somewhere since 1999, have an 18-game close-game stretch this good?
    # Estimate under each game's pregame odds.
    return dict(
        den_games=den[["game_id", "gameday", "margin", "spread_for", "wp_pre", "wp_close", "win"]].to_dict(orient="records"),
        den_record=[int(den.win.sum()), int(len(den) - den.win.sum())],
        den_expected_wins=float(den.wp_close.sum()),
        den_expected_wins_pregame=float(den.wp_pre.sum()),
        p_at_least=float((sims >= den.win.sum()).mean()),
        p_at_least_pregame=float((sims_pre >= den.win.sum()).mean()),
        close_fit=dict(const=float(fit.params["const"]), slope=float(fit.params["spread_for"])),
        sim_hist=np.bincount(sims, minlength=len(den) + 1).tolist(),
        league_best18=best.head(15).to_dict(orient="records"),
        n_teams_17plus=int((best.best18 >= 17).sum()),
        close_win_rate_by_spread=close.groupby(pd.cut(close.spread_for, [-30, -7, -3, 0, 3, 7, 30]), observed=True)
            .agg(win=("win", "mean"), wp_pre=("wp_pre", "mean"), n=("win", "size")).reset_index()
            .assign(spread_for=lambda x: x.spread_for.astype(str)).to_dict(orient="records"),
    )


def close_game_persistence():
    """Is winning close games a trait? Team-season close-game win% above expectation, year over year."""
    g = pd.read_csv(RAW / "games.csv")
    g = g[g.home_score.notna() & g.spread_line.notna() & (g.game_type == "REG")]
    rows = []
    for side in ["home", "away"]:
        m = np.where(side == "home", g.result, -g.result)
        s = np.where(side == "home", g.spread_line, -g.spread_line)
        rows.append(pd.DataFrame(dict(team=g[f"{side}_team"].values, season=g.season.values, margin=m, spread_for=s)))
    t = pd.concat(rows)
    t = t[(t.margin.abs() <= 8) & (t.margin != 0)]
    t["oe"] = (t.margin > 0) - spread_wp(t.spread_for)
    ts = t.groupby(["team", "season"]).agg(oe=("oe", "sum"), n=("oe", "size")).reset_index().sort_values(["team", "season"])
    ts["next"] = ts.groupby("team").oe.shift(-1)
    x = ts.dropna()
    return dict(yoy_corr=float(np.corrcoef(x.oe, x.next)[0, 1]), n=len(x),
                den=ts[ts.team == "DEN"].tail(4).to_dict(orient="records"))


def timeline(game_id):
    """Possession-by-possession: home win probability, and each drive's points over expected."""
    raw = pd.read_parquet(RAW / f"pbp_{int(game_id[:4])}.parquet",
                          columns=["game_id", "fixed_drive", "posteam", "home_team", "away_team", "qtr", "time",
                                   "vegas_home_wp", "home_wp", "total_home_score", "total_away_score", "desc", "play_type"])
    raw = raw[raw.game_id == game_id]
    d = pd.read_parquet(CLEAN / "nfl_drives_model.parquet")
    d = d[d.game_id == game_id][["fixed_drive", "posteam", "qtr", "start_half_secs", "points", "pts_hat", "poe", "result", "n_plays"]]
    first = raw.dropna(subset=["fixed_drive"]).groupby("fixed_drive").agg(
        wp=("vegas_home_wp", "first"), time=("time", "first"), hs=("total_home_score", "first"), as_=("total_away_score", "first"))
    d = d.merge(first, left_on="fixed_drive", right_index=True, how="left")
    plays = raw[raw.play_type.isin(["pass", "run", "punt", "field_goal", "kickoff", "extra_point"])][
        ["qtr", "time", "vegas_home_wp", "total_home_score", "total_away_score", "posteam", "desc"]]
    return dict(game_id=game_id, home=raw.home_team.iloc[0], away=raw.away_team.iloc[0],
                drives=d.to_dict(orient="records"),
                wp_series=plays.assign(i=np.arange(len(plays))).to_dict(orient="records"))


def main():
    p = pd.read_parquet(CLEAN / "nfl_plays_model.parquet")
    t, rel = team_seasons(p)
    q, qrel = qb_careers(p)
    out = dict(team_reliability=rel, qb_reliability=qrel,
               team_seasons=t.to_dict(orient="records"), qbs=q.to_dict(orient="records"),
               broncos=broncos_close_games(), close_persistence=close_game_persistence(),
               giants_broncos=timeline("2025_07_NYG_DEN"), sb47=timeline("2012_21_BAL_SF"))
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else round(float(o), 6)
        if isinstance(o, np.integer):
            return int(o)
        return o
    (OUT / "exploratory_nfl.json").write_text(json.dumps(clean(out)))
    print("team", rel, "\nqb", qrel)
    print(t.sort_values("excess").tail(5)[["posteam", "season", "excess", "z", "level"]])
    print(t.sort_values("excess").head(5)[["posteam", "season", "excess", "z", "level"]])
    print(q.sort_values("z")[["passer_player_name", "games", "excess", "z"]].head(5))
    print(q.sort_values("z")[["passer_player_name", "games", "excess", "z"]].tail(5))
    b = out["broncos"]
    print("DEN", b["den_record"], round(b["den_expected_wins"], 2), b["p_at_least"], "teams 17+:", b["n_teams_17plus"])
    print(pd.DataFrame(b["league_best18"]).head(8))
    print("close persistence", out["close_persistence"]["yoy_corr"])


if __name__ == "__main__":
    main()
