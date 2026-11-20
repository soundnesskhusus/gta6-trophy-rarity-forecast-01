"""GTA VI trophy rarity forecaster, round 2.

Forecasts every trophy's earn rate on Exophase at a case's target, from the
reads in the case's own history. The forecast is built from an earners /
players decomposition: the number of tracked players who have earned a trophy
and the number of tracked players are modelled separately, then divided.

Three extrapolators are combined per trophy (see forecast_trophy):
  * a log-time line in logit space (rate against ln(1 + hours since release)),
  * a recent exponential run on earners, damped by an archetype ceiling,
  * a saturating growth curve on earners (added in a later step).
Their weights come from rolling-origin backtests inside the case's history,
so a variant that has actually predicted this game's recent reads gets more
say than one that has not. Quantile spreads are calibrated from the same
backtests, aiming for honest ~90% coverage instead of the over-wide bands
that cost pinball loss at every trophy.

A stale page (reads unchanged for hours) means Exophase has not refreshed
its counts: the answer will be a refreshed page, so the forecast dilutes the
rates by the players jump a refresh brings, while letting earners keep
growing. The dilution size is read from the history's own past refresh
jumps when there are any, else from the launch-night jump the rules
describe (638 -> 3,868 players).

Only the case file in the input directory is read; there is no network at
grade time. Python 3.13 standard library only, fully deterministic.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Release of the game, the origin of every "hours since release" clock.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# The players jump of the launch-night refresh the rules describe (638 -> 3868),
# used as the dilution prior only when the history shows no refresh of its own.
LAUNCH_DILUTION = 3868.0 / 638.0


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    p = clamp_pct(x) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, halves away from zero (the kit's convention)."""
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(moment):
    return (moment - RELEASE).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# Archetypes: what kind of trophy this is, from the case's own metadata.
# The archetype sets the priors a thin history cannot: how high a rate can
# eventually go, how fast earners arrive, and how wide the bands should be.
# ---------------------------------------------------------------------------

def archetype(trophy, last_rate):
    if trophy["tier"] == "platinum":
        return "platinum"
    if last_rate is not None and last_rate >= 99.99:
        return "universal"
    text = (trophy["name"] + " " + trophy["description"]).lower()
    if "chapter" in text or "final mission" in text or trophy["tier"] == "gold":
        return "story"
    if any(k in text for k in ("stunt", "challenge", "collectible", "vehicle",
                               "completion", "100%")):
        return "grind"
    return "activity"


# Eventual ceiling: the share of tracked players who could ever earn a trophy
# of this kind (fraction, not percent). Story ceilings fall with depth: the
# first chapter is finished by most players, later ones by fewer.
ARCH_CEILING = {
    "story": 0.80,
    "activity": 0.35,
    "grind": 0.10,
    "platinum": 0.02,
    "universal": 1.0,
}
# A floor rate (percent) for a trophy nobody has earned yet: someone will
# have earned it by the target, and 0.00% would score badly if they have.
ARCH_FLOOR_PCT = {
    "story": 0.05,
    "activity": 0.02,
    "grind": 0.01,
    "platinum": 0.005,
    "universal": 100.0,
}
# Saturating-growth shape the earners curve starts from when the data is
# thin: (tau hours, shape k). Story tau grows with depth into the story.
ARCH_WEIBULL = {
    "story": (72.0, 1.5),
    "activity": (36.0, 1.2),
    "grind": (240.0, 1.0),
    "platinum": (720.0, 1.3),
    "universal": (24.0, 1.0),
}


def story_depth(trophies, idx):
    """How deep into the story this trophy sits (0 = earliest story trophy)."""
    depth = 0
    for j, t in enumerate(trophies):
        if j == idx:
            return depth
        if archetype(t, None) == "story":
            depth += 1
    return depth


def ceiling_for(trophies, idx, kind):
    if kind == "story":
        return ARCH_CEILING["story"] * (0.85 ** story_depth(trophies, idx))
    return ARCH_CEILING[kind]


# ---------------------------------------------------------------------------
# History, unpacked once per case.
# ---------------------------------------------------------------------------

class History:
    def __init__(self, case):
        self.times = []      # hours since release, per read
        self.players = []    # players tracked, per read
        self.rates = []      # rates[i][j]: trophy i, read j (None where missing)
        self.earned = []     # earners[i][j] = rates * players / 100
        n = len(case["trophies"])
        self.rates = [[] for _ in range(n)]
        self.earned = [[] for _ in range(n)]
        for read in case["history"]:
            moment = parse_time(read["at"])
            self.times.append(hours_since_release(moment))
            p = float(read["players_tracked"])
            self.players.append(p)
            for i in range(n):
                r = read["rates"][i]
                self.rates[i].append(r)
                self.earned[i].append(None if r is None else r * p / 100.0)
        self.n_reads = len(self.times)
        self.last_players = self.players[-1] if self.players else 0.0
        self.last_rates = [self.rates[i][-1] if self.rates[i] else None
                           for i in range(n)]

    def recent(self, hours, upto=None):
        """Indices of reads in the last `hours` before read `upto` (default: last)."""
        end = self.n_reads - 1 if upto is None else upto
        t_end = self.times[end]
        return [j for j in range(end + 1) if t_end - self.times[j] <= hours + 1e-9]


def fit_line(xs, ys):
    """Least squares y = a + b*x. Returns (a, b) or None when degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def players_forecast(hist, h_target):
    """Tracked players at the target hour, from a log-time fit.

    log P is fitted against ln(1 + hours since release) over the last 48h of
    reads: early on, players arrive roughly like a power of time. Returns
    (players, log_growth_per_day) for the spread model.
    """
    idx = hist.recent(48.0)
    if len(idx) < 2:
        idx = list(range(hist.n_reads))
    xs = [math.log1p(hist.times[j]) for j in idx]
    ys = [math.log(max(1.0, hist.players[j])) for j in idx]
    fit = fit_line(xs, ys)
    if fit is None:
        return hist.last_players, 0.0
    a, b = fit
    log_p = a + b * math.log1p(h_target)
    # The same fit's implied daily log growth at the target hour.
    daily = b * (24.0 / (1.0 + h_target))
    return max(hist.last_players, math.exp(log_p)), daily


def variant_logtime(hist, i, h_target, upto=None):
    """Logit rate linear in ln(1 + hours), fitted on the last 24h of reads.

    Returns a median rate percent, or None when the trophy has no usable
    positive rate in the window.
    """
    end = hist.n_reads - 1 if upto is None else upto
    idx = [j for j in hist.recent(24.0, upto=end)
           if hist.rates[i][j] is not None and 0.0 < hist.rates[i][j] < 100.0]
    if len(idx) < 3:
        idx = [j for j in range(end + 1)
               if hist.rates[i][j] is not None and 0.0 < hist.rates[i][j] < 100.0]
    if len(idx) < 2:
        return None
    xs = [math.log1p(hist.times[j]) for j in idx]
    ys = [logit_pct(hist.rates[i][j]) for j in idx]
    fit = fit_line(xs, ys)
    if fit is None:
        return None
    a, b = fit
    return clamp_pct(inv_logit_pct(a + b * math.log1p(h_target)))


def variant_recent_exp(hist, i, h_target, p_target, ceiling, upto=None):
    """Recent exponential run on earners, damped by the archetype ceiling.

    ln(earners) is fitted against clock hours over the last 12h, then the
    raw extrapolation is pulled back towards the ceiling with a soft cap, so
    a hot streak cannot forecast more earners than the player base can hold.
    Returns a median rate percent, or None when earners are all zero.
    """
    end = hist.n_reads - 1 if upto is None else upto
    idx = [j for j in hist.recent(12.0, upto=end)
           if hist.earned[i][j] is not None and hist.earned[i][j] > 0]
    if len(idx) < 3:
        idx = [j for j in range(end + 1)
               if hist.earned[i][j] is not None and hist.earned[i][j] > 0]
    if len(idx) < 2:
        return None
    xs = [hist.times[j] for j in idx]
    ys = [math.log(hist.earned[i][j]) for j in idx]
    fit = fit_line(xs, ys)
    if fit is None:
        return None
    a, g = fit
    raw = math.exp(a + g * h_target)
    cap_earners = max(1.0, p_target * ceiling)
    damped = raw / (1.0 + raw / cap_earners)
    return clamp_pct(100.0 * damped / max(1.0, p_target))


WEIBULL_TAUS = (8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0)
WEIBULL_KS = (0.8, 1.2, 1.8, 2.6)


def variant_weibull(hist, i, h_target, p_target, tau0, k0, ceiling=1.0, upto=None):
    """Saturating growth on earners: E(h) = S * (1 - exp(-(h/tau)^k)).

    The shape (tau, k) is chosen from a grid. S is solved in closed form
    for each shape and the shape is picked by a variance-stabilised error,
    so the fit is stable on counts that run from zero to thousands. The
    grid stops at timescales the history can still speak about: with two
    days of reads, a still-growing trophy and a saturating one look alike,
    and a shape that runs away to a power law would only invent growth.
    When earners have been flat, the shape is unidentified and the forecast
    stays flat instead of inventing growth from the clock alone. Two days
    of history also cannot tell a still-growing trophy from a saturating
    one: many shapes fit almost equally well while extrapolating very
    differently. Among shapes within 10% of the best fit, the forecast
    takes the median extrapolation, not the most extreme: in launch week
    growth continuing is common, but the fastest-growing shape is usually
    an artefact of the grid's edge. The rate is also capped at the
    archetype ceiling: no trophy can pass the share of players who could
    ever earn it. Returns a median rate percent, or None when the trophy
    has no earners at all in the window.
    """
    end = hist.n_reads - 1 if upto is None else upto
    idx = [j for j in range(end + 1) if hist.earned[i][j] is not None]
    if not idx or all(hist.earned[i][j] == 0 for j in idx):
        return None
    hs = [hist.times[j] for j in idx]
    es = [hist.earned[i][j] for j in idx]
    e_mean = sum(es) / len(es)
    if max(es) - min(es) <= 0.05 * (1.0 + e_mean):
        return clamp_pct(100.0 * es[-1] / max(1.0, p_target))
    scored = []
    for tau in WEIBULL_TAUS:
        for k in WEIBULL_KS:
            w = [1.0 - math.exp(-((h / tau) ** k)) if h > 0 else 0.0 for h in hs]
            sw2 = sum(v * v for v in w)
            if sw2 <= 1e-12:
                continue
            s = sum(e * v for e, v in zip(es, w)) / sw2
            if s <= 0:
                continue
            sse = sum((e - s * v) ** 2 / (1.0 + e) for e, v in zip(es, w))
            e_t = s * (1.0 - math.exp(-((h_target / tau) ** k))) if h_target > 0 else 0.0
            scored.append((sse, e_t))
    if not scored:
        return None
    best_sse = min(s for s, _ in scored)
    # Tie-break: two days of history cannot tell a still-growing trophy
    # from a saturating one -- many shapes fit almost equally well while
    # extrapolating very differently. Among shapes within 10% of the best
    # fit, take the median extrapolation: the most extreme shape is
    # usually an artefact of the grid's edge, and the median hedges the
    # saturating and the still-growing readings of the same data.
    near = sorted(e_t for s, e_t in scored if s <= 1.10 * best_sse)
    e_target = near[len(near) // 2]
    rate = 100.0 * e_target / max(1.0, p_target)
    return clamp_pct(min(rate, 100.0 * ceiling))


def backtest_weights(hist, trophies, p_target_now):
    """Rolling-origin weights for the three variants, from the case itself.

    Pseudo-checkpoints are cut inside the history; each variant forecasts
    the rates at reads 6h, 24h and 36h later using only reads up to the cut,
    and is scored by its median's mean absolute logit error against the
    actual rates. The longest horizon carries half the weight: a 6h
    backtest cannot tell the variants apart, and it is the long horizon
    that decides the final -- a variant that fits the local acceleration
    can still overshoot badly four days out. Weights are a softmax over those errors, with
    a small uniform floor so no variant is ever fully excluded -- the
    hedge matters when the regime turns. Also returns, per archetype, the
    mean absolute logit error of the weight-blended median at the backtest
    horizons, which calibrates the final quantile bands.
    Returns (weights dict, resid_blend dict archetype -> mean |err|).
    """
    n = hist.n_reads
    if n < 8:
        return {"logtime": 1.0 / 3, "recent": 1.0 / 3, "weibull": 1.0 / 3}, {}
    span = hist.times[-1] - hist.times[0]
    if span < 12.0:
        return {"logtime": 1.0 / 3, "recent": 1.0 / 3, "weibull": 1.0 / 3}, {}
    cuts = sorted(set([int(n * 0.2), int(n * 0.4), int(n * 0.6)]))
    cuts = [c for c in cuts if 0 <= c < n - 1]
    horizons = [(6.0, 0.15), (24.0, 0.35), (36.0, 0.50)]
    errs = {"logtime": [], "recent": [], "weibull": []}
    points = []  # (archetype, actual rate, {variant: median}) for blend residuals
    for c in cuts:
        for horizon_h, _w in horizons:
            pt = None
            for j in range(c + 1, n):
                if hist.times[j] - hist.times[c] >= horizon_h - 1e-9:
                    pt = j
                    break
            if pt is None:
                continue
            h_pt = hist.times[pt]
            p_pt = hist.players[pt]
            for i, trophy in enumerate(trophies):
                actual = hist.rates[i][pt]
                if actual is None or actual <= 0.0 or actual >= 100.0:
                    continue
                kind = archetype(trophy, hist.rates[i][c])
                if kind == "universal":
                    continue
                ceiling = ceiling_for(trophies, i, kind)
                tau0, k0 = ARCH_WEIBULL[kind]
                if kind == "story":
                    tau0 = 24.0 + 18.0 * story_depth(trophies, i)
                preds = {}
                v = variant_logtime(hist, i, h_pt, upto=c)
                if v is not None:
                    preds["logtime"] = v
                v = variant_recent_exp(hist, i, h_pt, p_pt, ceiling, upto=c)
                if v is not None:
                    preds["recent"] = v
                v = variant_weibull(hist, i, h_pt, p_pt, tau0, k0, ceiling, upto=c)
                if v is not None:
                    preds["weibull"] = v
                if not preds:
                    continue
                for name, med in preds.items():
                    e = abs(logit_pct(actual) - logit_pct(med))
                    errs[name].append((e, _w))
                points.append((kind, actual, dict(preds)))
    def wmean(pairs):
        tot = sum(w for _, w in pairs)
        return sum(e * w for e, w in pairs) / tot if tot > 0 else None
    means = {k: wmean(v) for k, v in errs.items()}
    present = {k: v for k, v in means.items() if v is not None}
    if not present:
        return {"logtime": 1.0 / 3, "recent": 1.0 / 3, "weibull": 1.0 / 3}, {}
    if max(present.values()) - min(present.values()) < 1e-9:
        weights = {k: 1.0 / 3 for k in ("logtime", "recent", "weibull")}
    else:
        raw = {k: math.exp(-v / 0.08) for k, v in present.items()}
        tot = sum(raw.values())
        weights = {}
        for k in ("logtime", "recent", "weibull"):
            soft = raw.get(k, 0.0) / tot if tot > 0 else 0.0
            weights[k] = 0.75 * soft + 0.25 / 3.0
        tot2 = sum(weights.values())
        weights = {k: v / tot2 for k, v in weights.items()}
    resid_blend = {}
    if points:
        by_kind = {}
        for kind, actual, preds in points:
            wsum = sum(weights[k] for k in preds)
            blend = sum(weights[k] * logit_pct(p) for k, p in preds.items()) / wsum
            by_kind.setdefault(kind, []).append(abs(logit_pct(actual) - blend))
        for kind, xs in by_kind.items():
            if len(xs) >= 5:
                resid_blend[kind] = sum(xs) / len(xs)
    return weights, resid_blend


def staleness(hist):
    """Is the page stale -- reads unchanged for hours -- and by how much
    would a refresh dilute the rates?

    Exophase refreshes a game's counts on its own schedule, hours apart,
    not continuously: a long run of identical reads means the page is
    showing old counts and the answer will be a refreshed page. The
    dilution factor is the median players jump the history itself has
    seen at a refresh, or the launch-night jump the rules describe when
    the history shows none. Returns (stale_hours, dilution, run_start_hour)
    or None.
    """
    n = hist.n_reads
    if n < 2:
        return None
    run_start = n - 1
    while run_start > 0:
        j = run_start - 1
        same_players = hist.players[j] == hist.players[run_start]
        same_rates = all(hist.rates[i][j] == hist.rates[i][run_start]
                         for i in range(len(hist.rates)))
        if same_players and same_rates:
            run_start = j
        else:
            break
    stale_hours = hist.times[-1] - hist.times[run_start]
    if stale_hours < 2.0:
        return None
    jumps = []
    for j in range(1, n):
        if hist.players[j - 1] > 0 and hist.players[j] > 1.05 * hist.players[j - 1]:
            jumps.append(hist.players[j] / hist.players[j - 1])
    if jumps:
        jumps.sort()
        dilution = jumps[len(jumps) // 2]
    else:
        dilution = LAUNCH_DILUTION
    return stale_hours, dilution, hist.times[run_start]


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    hist = History(case)
    h_target = hours_since_release(parse_time(case["target"]))
    horizon_hours = case["horizon_minutes"] / 60.0
    p_target, _ = players_forecast(hist, h_target)
    weights, resid_blend = backtest_weights(hist, case["trophies"], p_target)
    # Earners-based variants compound two extrapolations (earners and
    # players) while the rate-space variant extrapolates one quantity, so
    # their edge fades with horizon. Penalise them mildly with horizon.
    e_factor = 1.0 / math.sqrt(1.0 + horizon_hours / 24.0)
    adj = dict(weights)
    adj["recent"] *= e_factor
    adj["weibull"] *= e_factor
    wtot = sum(adj.values())
    weights = {k: v / wtot for k, v in adj.items()}
    # The rate-space variant estimates the ratio directly, so its two
    # error sources cancel instead of compounding: it is structurally the
    # most reliable at a long horizon, and the backtests -- which only
    # reach 36h -- cannot be trusted to demote it below three quarters.
    # The earners variants keep a quarter as a hedge for regimes where
    # the ratio's trend breaks (saturation, refresh steps).
    if weights["logtime"] < 0.75:
        rest = 0.25
        e_tot = weights["recent"] + weights["weibull"]
        weights = {"logtime": 0.75,
                   "recent": rest * weights["recent"] / e_tot if e_tot else 0.0,
                   "weibull": rest * weights["weibull"] / e_tot if e_tot else 0.0}
    stale = staleness(hist)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        kind = archetype(trophy, hist.last_rates[i])
        if kind == "universal":
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        ceiling = ceiling_for(case["trophies"], i, kind)
        tau0, k0 = ARCH_WEIBULL[kind]
        if kind == "story":
            tau0 = 24.0 + 18.0 * story_depth(case["trophies"], i)
        # On a long horizon this branch does not apply: 4.5 days of growth
        # swamps one refresh, and the ensemble below is the right model.
        if stale is not None and horizon_hours <= 12.0:
            # The page is stale and the horizon is short: the answer is a
            # refreshed page. Players jump by the dilution factor while
            # earners grow by a factor between 1 and the dilution: the
            # honest middle is the geometric mean, scaled down for deeper
            # trophies that almost nobody can reach inside the window
            # (a first-chapter trophy gains many new earners in a few
            # hours; a final-mission trophy gains almost none).
            stale_hours, dilution, run_start_h = stale
            mid = math.sqrt(dilution)
            if kind == "story":
                decay = 0.6 ** story_depth(case["trophies"], i)
            elif kind == "activity":
                decay = 0.7
            elif kind == "grind":
                decay = 0.4
            else:  # platinum
                decay = 0.2
            growth = 1.0 + (mid - 1.0) * decay
            e_last = hist.earned[i][-1] or 0.0
            p_refreshed = hist.last_players * dilution
            median = 100.0 * e_last * growth / max(1.0, p_refreshed)
            median = max(median, ARCH_FLOOR_PCT[kind])
            sigma = math.sqrt(0.375 ** 2 + 0.45 ** 2)
            q = [round4(clamp_pct(inv_logit_pct(logit_pct(median) + z * sigma)))
                 for z in Z]
            rows.append({"id": trophy["id"], "q": q})
            continue
        preds = {}
        v2 = variant_logtime(hist, i, h_target)
        if v2 is not None:
            preds["logtime"] = v2
        v3 = variant_recent_exp(hist, i, h_target, p_target, ceiling)
        if v3 is not None:
            preds["recent"] = v3
        v1 = variant_weibull(hist, i, h_target, p_target, tau0, k0, ceiling)
        if v1 is not None:
            preds["weibull"] = v1
        if preds:
            wsum = sum(weights[k] for k in preds)
            med_logit = sum(weights[k] * logit_pct(p) for k, p in preds.items()) / wsum
            median = inv_logit_pct(med_logit)
        else:
            median = ARCH_FLOOR_PCT[kind]
        median = max(median, ARCH_FLOOR_PCT[kind])
        # Spread: the blended backtest residual (a ~27h-horizon error
        # scale) grown to the real horizon, with the growth capped: a
        # trend fit's error grows much slower than a random walk's, so an
        # uncapped square-root growth would only widen the bands past
        # honesty. Never below the backtest scale, never above a wide band.
        # A flat tail on a long horizon adds refresh uncertainty on top.
        base = resid_blend.get(kind, 0.30)
        growth = min(math.sqrt(max(horizon_hours, 27.0) / 27.0), 1.8)
        sigma = 1.25 * base * growth
        if stale is not None:
            sigma = math.sqrt(sigma ** 2 + 0.20 ** 2)
        sigma = min(1.30, max(0.30, sigma))
        q = [round4(clamp_pct(inv_logit_pct(logit_pct(median) + z * sigma)))
             for z in Z]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
