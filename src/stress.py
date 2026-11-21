"""Stress tests, run after the pre-registered tests (so: labeled post hoc, all NFL seasons).

1. Split the 'situation' piece into down/distance/field vs score/clock/strength (the rubber band).
2. Try to break the null: minimal controls with a whole-game shuffle, EPA instead of success,
   live games only, streaks of 2 and 3, explosive plays, passes vs runs.
3. Forced pauses: does carryover survive a quarter break or a defensive timeout in the middle
   of a drive? The clock ends quarters whatever is happening, so a quarter break is a pause
   nobody chose.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from perm import group_codes, lag_pairs, within_group_perm, diff_stat, perm_test
from models import crossfit
from analysis import multi_mask_perm, excess_of, contrast

ROOT = Path(__file__).resolve().parents[1]
CLEAN, RAW, OUT = ROOT / "data" / "clean", ROOT / "data" / "raw" / "nfl", ROOT / "results"
MIN_FEATURES = ["down", "ydstogo", "yardline_100", "goal_to_go", "first_play_of_drive", "pass", "season"]


def gap_flags(seasons):
    """For each scrimmage play: what happened since the previous scrimmage play in the game."""
    out = []
    for s in seasons:
        d = pd.read_parquet(RAW / f"pbp_{s}.parquet", columns=["game_id", "play_id", "order_sequence", "qtr", "play_type",
                                                               "qb_kneel", "qb_spike", "two_point_attempt", "down", "epa",
                                                               "posteam", "timeout", "timeout_team", "desc", "half_seconds_remaining"])
        d = d.sort_values(["game_id", "order_sequence", "play_id"]).reset_index(drop=True)
        scr = (d.play_type.isin(["pass", "run"]) & (d.qb_kneel != 1) & (d.qb_spike != 1) & (d.two_point_attempt != 1)
               & d.down.notna() & d.epa.notna() & d.posteam.notna() & (d.qtr <= 4))
        d["k"] = scr.astype(int).groupby(d.game_id).cumsum()  # rows with k = n sit after the n-th scrimmage play
        d["gap"] = np.where(scr, d.k - 1, d.k)               # the gap before scrimmage play number gap+1
        d["qend"] = d.desc.str.contains("END QUARTER 1|END QUARTER 3", na=False, regex=True)
        d["two_min"] = d.desc.str.contains("Two-Minute Warning", case=False, na=False)
        g = d[~scr].groupby(["game_id", "gap"]).agg(
            to_team=("timeout_team", lambda x: ",".join(x.dropna().astype(str))), qend=("qend", "any"), two_min=("two_min", "any"))
        p = d[scr][["game_id", "play_id", "posteam", "k"]].copy()
        p["gap"] = p.k - 1
        p = p.merge(g, on=["game_id", "gap"], how="left")
        p["def_timeout"] = [(isinstance(t, str) and any(x and x != o for x in t.split(","))) for t, o in zip(p.to_team, p.posteam)]
        p["off_timeout"] = [(isinstance(t, str) and o in t.split(",")) for t, o in zip(p.to_team, p.posteam)]
        p["qend"] = p.qend.fillna(False).astype(bool)
        p["two_min"] = p.two_min.fillna(False).astype(bool)
        out.append(p[["game_id", "play_id", "def_timeout", "off_timeout", "qend", "two_min"]])
    return pd.concat(out, ignore_index=True)


def streak_stat(g, y, r, k, n_perm=1000, seed=31):
    """Mean residual after k straight successes minus after k straight failures (same unit, same half)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(k, len(g))
    ok = np.all([g[idx] == g[idx - j] for j in range(1, k + 1)], axis=0)
    t = idx[ok]

    def f(yy, rr):
        prev = np.stack([yy[t - j] for j in range(1, k + 1)])
        hot, cold = prev.all(0), (~prev.astype(bool)).all(0)
        return rr[t][hot].mean() - rr[t][cold].mean()
    obs = f(y, r)
    null = np.array([f(y[i], r[i]) for i in (within_group_perm(g, rng) for _ in range(n_perm))])
    mu, sd = null.mean(), null.std(ddof=1)
    return dict(obs=obs, null_mean=mu, excess=obs - mu, null_sd=sd, z=(obs - mu) / sd,
                p=(1 + np.sum(np.abs(null - mu) >= abs(obs - mu))) / (n_perm + 1), n=int(ok.sum()))


def main(n_perm=1000):
    p = pd.read_parquet(CLEAN / "nfl_plays_model.parquet")
    p["r_min"] = p.success - crossfit(p, MIN_FEATURES, "success", "clf")
    flags = gap_flags(sorted(p.season.unique()))
    p = p.merge(flags, on=["game_id", "play_id"], how="left")
    for c in ["def_timeout", "off_timeout", "qend", "two_min"]:
        p[c] = p[c].fillna(False).astype(bool)
    p = p.sort_values(["game_id", "posteam", "half", "seq"]).reset_index(drop=True)
    S = {}

    # 1. decomposition on the pre-registered holdout
    h = p[p.season.between(2019, 2025)].reset_index(drop=True)
    g = group_codes(h, ["game_id", "posteam", "half"])
    a, b = lag_pairs(g)
    y = h.success.values
    raw = y[b][y[a] == 1].mean() - y[b][y[a] == 0].mean()
    after_min = diff_stat(y[a], h.r_min.values[b])
    after_full = diff_stat(y[a], h.r.values[b])
    res = perm_test(g, y, h.r.values, n_perm=n_perm)
    S["decomposition"] = dict(raw=raw * 100, down_distance=(raw - after_min) * 100, score_clock_strength=(after_min - after_full) * 100,
                              good_half=res["null_mean"] * 100, order=res["excess"] * 100, order_sd=res["null_sd"] * 100)

    # 2. try to break it (all seasons 1999-2026)
    G = group_codes(p, ["game_id", "posteam", "half"])
    Gg = group_codes(p.sort_values(["game_id", "posteam", "seq"]).reset_index(drop=True), ["game_id", "posteam"])
    Y, R, Re = p.success.values, p.r.values, p.r_epa.values
    tries = []
    pg = p.sort_values(["game_id", "posteam", "seq"]).reset_index(drop=True)
    x = perm_test(group_codes(pg, ["game_id", "posteam"]), pg.success.values, pg.r_min.values, n_perm=n_perm)
    tries.append(("Bare-bones controls (down, distance, field) and a whole-game shuffle", x["excess"] * 100, x["null_sd"] * 100, x["p"], "pts"))
    x = perm_test(G, Y, R, n_perm=n_perm)
    tries.append(("Main test, all 28 seasons", x["excess"] * 100, x["null_sd"] * 100, x["p"], "pts"))
    x = perm_test(G, Y, Re, n_perm=n_perm)
    tries.append(("Next-play EPA instead of success", x["excess"], x["null_sd"], x["p"], "EPA"))
    live = p[(p.wp >= .1) & (p.wp <= .9)].reset_index(drop=True)
    x = perm_test(group_codes(live, ["game_id", "posteam", "half"]), live.success.values, live.r.values, n_perm=n_perm)
    tries.append(("Live games only (win chance 10–90%)", x["excess"] * 100, x["null_sd"] * 100, x["p"], "pts"))
    for k in [2, 3]:
        x = streak_stat(G, Y, R, k, n_perm=n_perm)
        tries.append((f"After {k} straight successes vs {k} straight failures", x["excess"] * 100, x["null_sd"] * 100, x["p"], "pts"))
    a, b = lag_pairs(G)
    big = (p.epa.values >= 2).astype(int)
    rng = np.random.default_rng(41)
    def bigstat(bb, yy, rr):
        m1, m0 = bb[a] == 1, yy[a] == 0
        return rr[b][m1].mean() - rr[b][m0].mean()
    obs = bigstat(big, Y, R)
    null = []
    for _ in range(n_perm):
        i = within_group_perm(G, rng); null.append(bigstat(big[i], Y[i], R[i]))
    null = np.array(null)
    tries.append(("After a big play (2+ EPA) vs after a failure", (obs - null.mean()) * 100, null.std(ddof=1) * 100,
                  (1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (n_perm + 1), "pts"))
    cur_pass = p["pass"].values[b] == 1
    obs_, null_, _ = multi_mask_perm(G, Y, R, {"pass": cur_pass, "run": ~cur_pass}, n_perm=n_perm)
    for k_, lab in [("pass", "Next play is a pass"), ("run", "Next play is a run")]:
        x = excess_of(obs_, null_, k_)
        tries.append((lab, x["excess"] * 100, x["null_sd"] * 100, x["p"], "pts"))
    S["tries"] = [dict(label=l, est=e, sd=s, p=pp, unit=u) for l, e, s, pp, u in tries]

    # 3. forced pauses inside a drive
    same = p.fixed_drive.values[a] == p.fixed_drive.values[b]
    late = (p.half_seconds_remaining.values[b] <= 120) | p.two_min.values[b]
    qend = p.qend.values[b] & same
    dto = p.def_timeout.values[b] & same & ~late & ~qend
    clean_ = same & ~p.qend.values[b] & ~p.def_timeout.values[b] & ~p.off_timeout.values[b] & ~late
    obs_, null_, _ = multi_mask_perm(G, Y, R, {"none": clean_, "qend": qend, "dto": dto}, n_perm=2000)
    S["pauses"] = {k: dict(excess_of(obs_, null_, k), n=int(m.sum())) for k, m in {"none": clean_, "qend": qend, "dto": dto}.items()}
    S["pauses"]["none_minus_qend"] = contrast(obs_, null_, {"none": 1, "qend": -1})
    S["pauses"]["none_minus_dto"] = contrast(obs_, null_, {"none": 1, "dto": -1})
    # Does a timeout cool a hot offense? Mean next-play residual after 2 straight successes in the drive.
    i = np.arange(2, len(G))
    hot = (G[i] == G[i - 1]) & (G[i] == G[i - 2]) & (Y[i - 1] == 1) & (Y[i - 2] == 1) & \
          (p.fixed_drive.values[i] == p.fixed_drive.values[i - 2]) & (p.half_seconds_remaining.values[i] > 120) & ~p.two_min.values[i]
    ti = p.def_timeout.values[i] & ~p.qend.values[i]
    r_ice, r_no = R[i][hot & ti], R[i][hot & ~ti & ~p.off_timeout.values[i] & ~p.qend.values[i]]
    games = p.game_id.values[i]
    # game-clustered bootstrap for the difference
    rng = np.random.default_rng(5)
    gi = pd.factorize(games)[0]
    m_ice, m_no = hot & ti, hot & ~ti & ~p.off_timeout.values[i] & ~p.qend.values[i]
    ng = gi.max() + 1
    s_ice = np.bincount(gi[m_ice], R[i][m_ice], ng); c_ice = np.bincount(gi[m_ice], minlength=ng)
    s_no = np.bincount(gi[m_no], R[i][m_no], ng); c_no = np.bincount(gi[m_no], minlength=ng)
    boots = []
    for _ in range(2000):
        w = np.bincount(rng.integers(0, ng, ng), minlength=ng)
        boots.append((w @ s_ice) / max(w @ c_ice, 1) - (w @ s_no) / max(w @ c_no, 1))
    est = r_ice.mean() - r_no.mean()
    S["ice"] = dict(est=est * 100, ci=[float(np.percentile(boots, 2.5)) * 100, float(np.percentile(boots, 97.5)) * 100],
                    n_ice=int(m_ice.sum()), n_no=int(m_no.sum()), mean_ice=r_ice.mean() * 100, mean_no=r_no.mean() * 100)

    # 4. yardstick for the funnel: how much does a real team trait repeat year to year?
    ts = p.groupby(["posteam", "season"]).success.mean().reset_index().sort_values(["posteam", "season"])
    ts["next"] = ts.groupby("posteam").success.shift(-1)
    x = ts.dropna()
    S["offense_yoy"] = float(np.corrcoef(x.success, x["next"])[0, 1])

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if k != "null"}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, np.bool_)):
            return int(o)
        return o
    (OUT / "stress_nfl.json").write_text(json.dumps(clean(S), indent=1))
    print(json.dumps(clean(S), indent=1)[:4000])


if __name__ == "__main__":
    main()
