"""Collect every number the interactive page shows into docs/data.json, straight from
results/*.json and the simulation output, so the page cannot drift from the analysis."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def r(x, k=5):
    if isinstance(x, dict):
        return {a: r(b, k) for a, b in x.items()}
    if isinstance(x, list):
        return [r(b, k) for b in x]
    if isinstance(x, float):
        return round(x, k)
    return x


def load(name):
    f = RES / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def ci(e, sd):
    return [e - 1.96 * sd, e + 1.96 * sd]


def tests(R, league):
    if R is None:
        return None
    out = {}
    def perm(key, label, unit, scale, direction, kind="primary", note=""):
        x = R[key]
        out[key] = dict(label=label, unit=unit, kind=kind, direction=direction, note=note,
                        est=x["excess"] * scale, ci=[v * scale for v in ci(x["excess"], x["null_sd"])],
                        mde=x["mde80"] * scale, p=x["p"], holm=R["holm"].get(key))
    perm("P1", "Play → next play", "pts", 100, "+", note="Next-play success after a success vs after a failure, beyond the shuffle.")
    perm("P2", "Drive → our next drive", "pts per pt", 1, "+", note="Points over expected on the next drive, per point over expected on this one.")
    perm("P3", "Our drive → their next drive", "pts per pt", 1, "−", note="Momentum says a great drive makes our defense better next series: a negative slope.")
    x = R["P4"]
    out["P4"] = dict(label="Fumble bounce", unit="pts", kind="primary", direction="+",
                     note="Swing in the next two possessions when the bounce goes to the defense instead of the offense.",
                     est=x["est"], ci=list(x["ci"]), mde=x["mde80"], p=x["p_ri"], holm=R["holm"].get("P4"))
    bm = R["big_moments"]
    out["big"] = dict(label="After a 4th-down conversion", unit="pts", kind="secondary", direction="+",
                      note="Next play after a 4th-down conversion vs after a 3rd-down conversion.",
                      est=bm["excess"] * 100, ci=[v * 100 for v in ci(bm["excess"], bm["null_sd"])], mde=bm["mde80"] * 100, p=bm["p"])
    sg = R["stakes_crowd"]["stakes_gap"]
    out["stakes"] = dict(label="High stakes vs low stakes", unit="pts", kind="secondary", direction="+",
                         note="Carryover after the 20% of snaps with the most at stake minus the 20% with the least.",
                         est=sg["excess"] * 100, ci=[v * 100 for v in ci(sg["excess"], sg["null_sd"])], mde=2.8 * sg["null_sd"] * 100, p=sg["p"])
    ha = R["stakes_crowd"]["home_minus_away"]
    out["crowd"] = dict(label="Home offense vs away offense", unit="pts", kind="secondary", direction="+",
                        note="If the crowd feeds momentum, the home team should carry over more.",
                        est=ha["excess"] * 100, ci=[v * 100 for v in ci(ha["excess"], ha["null_sd"])], mde=2.8 * ha["null_sd"] * 100, p=ha["p"])
    if "crowd_2020" in R:
        c = R["crowd_2020"]["did"]
        out["empty"] = dict(label="2020 empty stadiums", unit="pts", kind="secondary", direction="+",
                            note="Home-minus-away carryover in 2020 minus 2019 and 2021. Crowd momentum predicts it drops.",
                            est=-c["excess"] * 100, ci=[-v * 100 for v in ci(c["excess"], c["null_sd"])][::-1], mde=2.8 * c["null_sd"] * 100, p=c["p"])
    b = R["breaks"]["none_minus_halftime"]
    out["halftime"] = dict(label="Halftime reset", unit="pts per pt", kind="secondary", direction="+",
                           note="Drive carryover with no break minus across halftime. A reset predicts positive.",
                           est=b["excess"], ci=ci(b["excess"], b["null_sd"]), mde=2.8 * b["null_sd"], p=b["p"])
    for k in ["P2_sr_sum", "P2_epa_oe"]:
        x = R[k]
        out[k] = dict(label="Drive → our next drive (" + ("successful plays" if "sr" in k else "EPA") + ")",
                      unit="per unit", kind="secondary", direction="+", note="The same drive-to-drive test, counting successful plays (or EPA) instead of points.",
                      est=x["excess"], ci=ci(x["excess"], x["null_sd"]), mde=x["mde80"], p=x["p"])
    return out


def waterfall(R):
    if R is None:
        return None
    w = R["waterfall"]
    return dict(raw=w["raw"] * 100, situation=(w["raw"] - w["after_situation"]) * 100, team=w["team_half_level"] * 100,
                order=w["order"] * 100, order_ci=[v * 100 for v in ci(w["order"], w["order_sd"])],
                after_success=w["p_after_success"] * 100, after_fail=w["p_after_fail"] * 100,
                n_pairs=w["n_pairs"], n_plays=R["n_plays"], n_games=R["n_games"], seasons=R["seasons"])


def main():
    splits = {k: load(k) for k in ["holdout_nfl", "discovery_nfl", "fresh_nfl", "holdout_cfb", "discovery_cfb"]}
    E = load("exploratory_nfl")
    sim = pd.read_parquet(ROOT / "data" / "clean" / "sim_results.parquet")
    sims = []
    for name, g in sim.groupby("setting", sort=False):
        sims.append(dict(setting=name, planted=float(g.planted.iloc[0]) * 100, raw=float(g.raw.mean() * 100),
                         est=[round(v * 100, 3) for v in g.excess], reject=float((g.p < 0.05).mean())))
    ts = pd.DataFrame(E["team_seasons"])[["posteam", "season", "excess", "null_sd", "z", "n_pairs", "level"]]
    ts[["excess", "null_sd", "level"]] *= 100
    qb = pd.DataFrame(E["qbs"])[["passer_player_name", "games", "excess", "null_sd", "z", "n_pairs", "level"]]
    qb[["excess", "null_sd", "level"]] *= 100

    def tl(T, keep_team):
        d = pd.DataFrame(T["drives"])
        d = d[["fixed_drive", "posteam", "qtr", "time", "hs", "as_", "points", "pts_hat", "poe", "result", "n_plays", "wp"]]
        w = pd.DataFrame(T["wp_series"])[["qtr", "time", "vegas_home_wp", "total_home_score", "total_away_score", "desc"]]
        w["blackout"] = w.desc.str.contains("power failure", na=False)
        w = w.drop(columns="desc")
        return dict(home=T["home"], away=T["away"], drives=r(d.to_dict(orient="records"), 3),
                    wp=r(w.to_dict(orient="records"), 3))

    data = dict(
        waterfall={k: waterfall(v) for k, v in splits.items()},
        tests={"holdout_nfl": tests(splits["holdout_nfl"], "nfl"), "discovery_nfl": tests(splits["discovery_nfl"], "nfl"),
               "holdout_cfb": tests(splits["holdout_cfb"], "cfb"), "fresh_nfl": tests(splits["fresh_nfl"], "nfl")},
        decay={k: [dict(lag=x["lag"], est=x["excess"] * 100, sd=x["null_sd"] * 100) for x in v["decay"]] for k, v in splits.items() if v},
        granularity={k: [dict(block=x["block"], est=x["excess"] * 100, sd=x["null_sd"] * 100) for x in v["granularity"]] for k, v in splits.items() if v},
        breaks={k: {b: dict(est=v["breaks"][b]["excess"], sd=v["breaks"][b]["null_sd"], n=v["breaks"][b].get("n")) for b in ["none", "quarter_break", "halftime"]} for k, v in splits.items() if v},
        sims=sims,
        team_seasons=r(ts.to_dict(orient="records"), 3), team_rel=E["team_reliability"],
        qbs=r(qb.to_dict(orient="records"), 3), qb_rel=E["qb_reliability"],
        broncos=E["broncos"], close_persistence=E["close_persistence"],
        giants=tl(E["giants_broncos"], "DEN"), sb47=tl(E["sb47"], "SF"),
        stress=load("stress_nfl"),
        qbs_small=r(pd.DataFrame(json.loads((RES / "qbs_small.json").read_text()))[["passer_player_name", "games", "excess", "null_sd", "z", "n_pairs", "level"]].assign(excess=lambda x: x.excess * 100, null_sd=lambda x: x.null_sd * 100, level=lambda x: x.level * 100).to_dict(orient="records"), 3),
    )
    (DOCS / "data.json").write_text(json.dumps(r(data, 4), separators=(",", ":")))
    print("data.json", round((DOCS / "data.json").stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
