"""Exploratory: 'hot at the right time.' Does a strong finish to the regular season predict
playoff results beyond what the point spread already knows? Seasons 1999-2025.

hot = mean(margin minus pregame spread) over a team's last 4 regular-season games.
       (Beating the spread means playing better than the market expected.)
Also: raw wins in the last 4, which is how fans and TV talk about it.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
g = pd.read_csv(ROOT / "data/raw/nfl/games.csv")
g = g[(g.season <= 2025) & g.home_score.notna() & g.spread_line.notna()].copy()
g = g.replace({"OAK": "LV", "SD": "LAC", "STL": "LA"})
rows = []
for side, opp, sgn in (("home", "away", 1), ("away", "home", -1)):
    t = pd.DataFrame(dict(season=g.season, week=g.week, game_type=g.game_type, game_id=g.game_id,
                          team=g[f"{side}_team"], margin=sgn * g.result, exp=sgn * g.spread_line))
    rows.append(t)
t = pd.concat(rows)
reg = t[t.game_type == "REG"].sort_values("week")
reg["beat"] = reg.margin - reg.exp
last4 = reg.groupby(["team", "season"]).tail(4).groupby(["team", "season"]).agg(
    hot=("beat", "mean"), last4_wins=("margin", lambda m: int((m > 0).sum())))
season = reg.groupby(["team", "season"]).agg(season_beat=("beat", "mean"))
form = last4.join(season).reset_index()

po = g[g.game_type != "REG"].copy()
for side in ("home", "away"):
    po = po.merge(form.add_prefix(side + "_").rename(columns={f"{side}_season": "season"}),
                  on=["season", f"{side}_team"], how="left")
po = po.dropna(subset=["home_hot", "away_hot"])
po["hot_diff"] = po.home_hot - po.away_hot
po["seasonbeat_diff"] = po.home_season_beat - po.away_season_beat
po["cover"] = po.result - po.spread_line           # home margin beyond the spread
po["home_win"] = (po.result > 0).astype(int)

def ols(y, X, name):
    m = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC1")
    return dict(coef=float(m.params[name]), lo=float(m.conf_int().loc[name, 0]),
                hi=float(m.conf_int().loc[name, 1]), p=float(m.pvalues[name]))

out = dict(games=len(po))
out["cover_on_hot"] = ols(po.cover, po[["hot_diff"]], "hot_diff")
out["cover_on_hot_ctrl_season"] = ols(po.cover, po[["hot_diff", "seasonbeat_diff"]], "hot_diff")
m0 = sm.Logit(po.home_win, sm.add_constant(po[["spread_line"]])).fit(disp=0)
m1 = sm.Logit(po.home_win, sm.add_constant(po[["spread_line", "hot_diff"]])).fit(disp=0)
out["win_on_hot_given_spread"] = dict(coef=float(m1.params.hot_diff), p=float(chi2.sf(2 * (m1.llf - m0.llf), 1)))

# fan version: team that won all of its last 4 vs one that didn't, against the spread
long = pd.concat([
    pd.DataFrame(dict(w4=po.home_last4_wins, beat=po.cover, win=po.home_win, exp=po.spread_line)),
    pd.DataFrame(dict(w4=po.away_last4_wins, beat=-po.cover, win=1 - po.home_win, exp=-po.spread_line))])
from scipy.stats import norm
long["exp_wp"] = norm.cdf(long.exp / 13.45)
fan = long.groupby(long.w4.clip(upper=4)).agg(n=("win", "size"), win_rate=("win", "mean"),
                                               expected=("exp_wp", "mean"), avg_vs_spread=("beat", "mean"))
out["by_last4_wins"] = fan.reset_index().to_dict(orient="records")
# champions' finishes
sbw = po[po.game_type == "SB"].assign(champ=lambda d: np.where(d.result > 0, d.home_team, d.away_team))
ch = form.merge(sbw[["season", "champ"]], left_on=["season", "team"], right_on=["season", "champ"])
out["champs_last4_wins"] = ch.last4_wins.value_counts().sort_index().to_dict()
out["champs_hot_mean"] = float(ch.hot.mean())
out["all_playoff_teams_hot_mean"] = float(long.assign(k=1).pipe(lambda x: x).shape[0] and form.merge(
    pd.concat([po[["season", "home_team"]].rename(columns={"home_team": "team"}),
               po[["season", "away_team"]].rename(columns={"away_team": "team"})]).drop_duplicates(),
    on=["season", "team"]).hot.mean())
clean = lambda o: {str(k): clean(v) for k, v in o.items()} if isinstance(o, dict) else [clean(v) for v in o] if isinstance(o, list) else round(float(o), 4) if isinstance(o, (float, np.floating)) else int(o) if isinstance(o, np.integer) else o
(ROOT / "results/hot_entering.json").write_text(json.dumps(clean(out), indent=1))
for k, v in out.items(): print(k, v)
