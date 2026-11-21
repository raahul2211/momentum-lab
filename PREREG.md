# Pre-registration: Can you find momentum?

Written and committed **before** any test was run on the holdout seasons. The git history
shows this file landing before `results/holdout_*.json` exists.

## Question

Momentum claims that a good result makes the *next* result more likely to be good, beyond what
the situation and the quality of the two teams already predict. Formally, for a unit's sequence of
outcomes y_t with situation-expected probability p_t:

  r_t = y_t − p_t
  r_t = α_block + ρ · r_{t−1} + ε_t

α_block is "better that day" (a level). ρ is momentum (order). The explainer's own test cannot
separate α from ρ over a quarter. We hold α fixed by shuffling inside small blocks and test ρ.

## Data and splits

| League | Source | Discovery (seen) | Holdout (unseen until this commit) |
|---|---|---|---|
| NFL | nflverse play-by-play | 1999–2018 | 2019–2025 (primary), 2026 weeks to date (fresh replication) |
| College | cfbfastR, FBS games with a betting line | 2014–2018 | 2019–2021 |

Plays: offensive pass or run, quarters 1–4, no kneels, spikes or two-point tries. Success = EPA > 0.
Expected success p_t: gradient-boosted model of situation (down, distance, field, clock, score,
timeouts, first play of drive, pass/run) plus pregame strength (spread, total, prior-games form)
plus environment. It never sees earlier plays from the same game. Cross-fitted by season.

Drives: points over expected (POE) = drive points minus a cross-fitted model of drive points from
the drive's starting state and pregame strength.

## Primary tests (two-sided, α = 0.05, Holm-adjusted within each league)

| ID | Test | Statistic | Null | Momentum predicts |
|---|---|---|---|---|
| P1 | Play → next play, same offense, same half | mean r_t after success − after failure | shuffle that offense's plays within the half, 2,000 times | > 0 |
| P2 | Drive → same offense's next drive, same half | slope of POE_{k+1} on POE_k, controlling for drive k seconds and plays | shuffle that offense's drives within the half (x and controls travel together) | > 0 |
| P3 | Drive → opponent's next drive (our defense), same half | slope of opponent POE on our POE, controlling for our drive's seconds and plays | Freedman–Lane residual permutation within team-game-half | < 0 |
| P4 | Fumble bounce (lost vs kept), live-ball scrimmage fumbles | next-possession swing = opponent next-drive POE − fumbling team next-drive POE, OLS with strata and pre-fumble covariates | randomization inference: shuffle "lost" within fumble type × forced | > 0 |

Reported for each: estimate, null mean, excess (estimate − null mean), z, p, and the smallest
effect the test could detect with 80% power (2.8 × null SD).

## Secondary tests (reported unadjusted, labeled secondary)

- P1 split into within-drive and across-drive pairs; lags 1–8 decay profile.
- Null granularity: whole game / half / quarter / drive blocks.
- P2–P4 with success-over-expected and EPA-over-expected drive sums.
- Stakes ("psychological mass", Adler 1981): carryover after top-20% vs bottom-20% pre-snap leverage.
- Big moments: next play after a 4th-down conversion vs after a 3rd-down conversion.
- Crowd: home-offense vs away-offense carryover; NFL 2020 (empty stadiums) difference-in-differences
  against 2019 and 2021.
- Breaks: same-offense drive carryover across no break, a quarter break, halftime.

## Exploratory (no claims of significance)

Team-season and quarterback "momentum index" (P1 excess per unit), funnel plots, year-over-year
stability, outliers. Case studies: Broncos games, Super Bowl XLVII blackout.

## What we already saw in discovery (disclosed)

NFL 1999–2018: P1 excess −0.03 pts (null SD 0.11); P2 excess +0.0017 points per point; P3 +0.0013;
P4 −0.04 points. A first P4 version using per-drive *average* residuals showed +3.2 pts (p < .001);
it vanished once drive outcomes were built as sums (an average over a drive is biased by drive
length, a sum is not). We keep sums.

College 2014–2018: the first pass ordered plays by file row, which is reversed for many ESPN games.
After ordering by ESPN sequence number: P1 excess −0.9 pts (z ≈ −6.5, anti-momentum), P2 +0.005
(p ≈ .10), P2 with success sums +0.010 (p ≈ .005), P3 −0.010 (p ≈ .02), P4 +0.04 points. College
fumble recovery fails the balance check (bounce depends on field position and win probability;
cfbfastR has no aborted-snap or out-of-bounds flags), so college P4 is reported but not interpreted
as a natural experiment.

## Validation run before the holdout

Simulated seasons on the real NFL situations (30 per setting): with team-half levels and no
momentum, P1 rejected 3/30; with planted momentum of +0.5 / +1 / +2 pts it rejected 80% / 100% /
100% and estimated 0.47 / 0.92 / 1.80. Mid-half level shifts (injury-style drift) cause false
positives (63%), so a positive P1 alone would not prove momentum; the decay profile and P3/P4 are
the checks for that.
