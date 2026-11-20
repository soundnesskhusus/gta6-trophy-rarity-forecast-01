[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/soundnesskhusus/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/soundnesskhusus/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecaster

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase
at one week after launch, as five quantiles per trophy. Entry for the
[make.fun bounty `gta6-trophy-rarity-forecast-01`](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

Run: `python3 main.py` (Python 3.13, standard library only, no network).
It reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and
writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

## Method

An earn rate is earners divided by players tracked, so the two are
modelled separately and divided again at the target:

- **Players tracked** follow a saturating curve, fitted on log players
  over a grid of timescales and extrapolated to the target.
- **Earners** per trophy follow a saturating Weibull-style curve. The
  (ceiling, timescale) pair is barely separable from two days of early
  data, so the entry averages over the whole fit grid with a likelihood
  times a prior: the prior's ceiling and timescale come from the trophy's
  archetype (story chapter, side mission, grind, easy activity, money,
  combat, flavour, platinum, 100% completion), inferred from its name,
  description and tier.
- The median is blended 80/20 with a log-time trend fit of the logit
  rate, as a hedge against the curve family being wrong.
- When the whole history is one frozen page (launch night), a refresh is
  overdue: the forecast applies a dilution drop (tracked players jump
  about sixfold while earners barely move).
- Quantile spreads are honest logit-scale widths that grow with the
  horizon and with how uncertain a trophy is (posterior spread of the
  earners average, archetype uncertainty, and disagreement between the
  two models).

## Validation

- Practice cases (reconstructed answers, calibrated so the house
  persistence baseline scores the ~1.51 the rules state): **0.21**
  pinball loss, vs 1.54 for the baseline.
- Synthetic final-regime backtests with known day-7 truth (three player
  growth scenarios): **0.19 / 0.21 / 0.21** vs 0.26 / 0.27 / 0.27 for a
  log-time trend baseline and 0.96 / 0.82 / 1.12 for persistence.
- Runtime ~0.05 s per practice case, ~0.08 s per full-history case
  (limit 8 s); two runs on the same input are byte-identical.
