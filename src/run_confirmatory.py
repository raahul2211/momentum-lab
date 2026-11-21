"""Run every pre-registered test for one league and one set of seasons; write JSON.

  python src/run_confirmatory.py nfl holdout     # 2019-2025
  python src/run_confirmatory.py nfl fresh       # 2026 to date
  python src/run_confirmatory.py nfl discovery   # 1999-2018
  python src/run_confirmatory.py cfb holdout     # 2019-2021
  python src/run_confirmatory.py cfb discovery   # 2014-2018
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis import (prep_plays, p1_play_carryover, play_waterfall, decay_profile, null_granularity,
                      leverage_and_crowd, crowd_2020, big_moments, drive_pairs, p2_same_unit, p3_cross_unit,
                      breaks_test, fumble_table, fumble_balance, p4_fumble, multi_mask_perm, excess_of)
from perm import holm, group_codes, lag_pairs

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "clean"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

SPLITS = {
    ("nfl", "discovery"): list(range(1999, 2019)),
    ("nfl", "holdout"): list(range(2019, 2026)),
    ("nfl", "fresh"): [2026],
    ("cfb", "discovery"): list(range(2014, 2019)),
    ("cfb", "holdout"): [2019, 2020, 2021],
}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if k != "null"}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main(league, split, n_perm=2000):
    t0 = time.time()
    seasons = SPLITS[(league, split)]
    p = pd.read_parquet(CLEAN / f"{league}_plays_model.parquet")
    d = pd.read_parquet(CLEAN / f"{league}_drives_model.parquet")
    f = pd.read_parquet(CLEAN / f"{league}_fumbles.parquet")
    pt = prep_plays(p, seasons)
    dt = drive_pairs(d, seasons)
    R = {"league": league, "split": split, "seasons": seasons, "n_plays": len(pt), "n_drives": len(dt),
         "n_games": int(pt.game_id.nunique())}

    # ---- primary
    R["P1"] = p1_play_carryover(pt, n_perm=n_perm)
    R["P2"] = p2_same_unit(dt, "poe", n_perm=n_perm)
    R["P3"] = p3_cross_unit(dt, "poe", n_perm=n_perm)
    ft = fumble_table(f, d, p, seasons)
    R["P4"] = p4_fumble(ft, "swing", n_perm=n_perm)
    pv = {"P1": R["P1"]["p"], "P2": R["P2"]["p"], "P3": R["P3"]["p"], "P4": R["P4"]["p_ri"]}
    R["holm"] = holm(pv)
    print("primary done", f"{time.time()-t0:.0f}s", {k: round(v, 4) for k, v in R["holm"].items()})

    # ---- secondary
    R["waterfall"] = play_waterfall(pt, n_perm=500)
    g = group_codes(pt, ["game_id", "posteam", "half"])
    a, b = lag_pairs(g)
    same = pt.fixed_drive.values[a] == pt.fixed_drive.values[b]
    obs, null, _ = multi_mask_perm(g, pt.success.values, pt.r.values, {"within": same, "across": ~same}, n_perm=1000)
    R["P1_within_drive"] = excess_of(obs, null, "within")
    R["P1_across_drive"] = excess_of(obs, null, "across")
    R["decay"] = decay_profile(pt, n_perm=300).to_dict(orient="records")
    R["granularity"] = null_granularity(pt, n_perm=500).to_dict(orient="records")
    R["stakes_crowd"] = leverage_and_crowd(pt, n_perm=1000)
    if league == "nfl" and split == "holdout":
        R["crowd_2020"] = crowd_2020(pt, n_perm=2000)
    R["big_moments"] = big_moments(pt, n_perm=1000)
    for o in ["sr_sum", "epa_oe"]:
        R[f"P2_{o}"] = p2_same_unit(dt, o, n_perm=1000)
        R[f"P3_{o}"] = p3_cross_unit(dt, o, n_perm=1000)
    R["breaks"] = breaks_test(dt, n_perm=1000)
    bal, m = fumble_balance(ft)
    R["fumble_balance"] = bal.to_dict(orient="records")
    R["fumble_balance_logit_p_form"] = {k: float(m.pvalues[k]) for k in ["a_form", "b_form"]}
    for o in ["swing_sr", "swing_epa", "rest_swing"]:
        R[f"P4_{o}"] = p4_fumble(ft, o, n_perm=1000)
    ft.to_parquet(CLEAN / f"{league}_{split}_fumble_table.parquet", index=False)
    R["runtime_s"] = time.time() - t0
    (OUT / f"{split}_{league}.json").write_text(json.dumps(clean(R), indent=1))
    print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
