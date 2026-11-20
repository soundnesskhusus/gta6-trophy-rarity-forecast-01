"""Forecast Grand Theft Auto VI (PS5) trophy earn rates at one week.

Reads every case file found in the input directory (schema
makefun/gta6-trophies-case/v1): a list of trophies plus the house's reads of
Exophase's trophy page up to the case checkpoint. Writes one forecast file
per case: for each trophy, five quantiles (5/25/50/75/95%) of its earn rate,
in percent, at the case target.

Method, in brief: an earn rate is earners divided by players tracked, so the
two are modelled separately. Players tracked follow a saturating growth
curve. Each trophy's earners follow a saturating Weibull-style curve whose
ceiling and timescale are fitted to the case history and shrunk toward a
prior set by the trophy's archetype (story chapter, side mission, grind,
easy activity, money, combat, flavour, platinum, 100% completion), which is
inferred from the trophy's name, description and tier. The two curves are
divided again at the target to give the median rate. When the whole history
is one frozen page (the launch-night stale page), a refresh is overdue and
the forecast instead applies a dilution drop. Spreads are honest logit-scale
widths that grow with the horizon and with how uncertain a trophy is.

Python 3.13, standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

# Normal quantiles for the five levels 0.05, 0.25, 0.5, 0.75, 0.95.
Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

# Earn rates are scored on the logit scale after clamping to this range.
LOGIT_LO = 0.01
LOGIT_HI = 99.99

# Release moment: midnight UTC on the game's release date.
RELEASE_ISO = "2026-11-19T00:00:00Z"


def parse_moment(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    moment = datetime.fromisoformat(text)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def hours_since_release(moment):
    """Hours from the release moment to the given moment (may be negative)."""
    origin = parse_moment(RELEASE_ISO)
    return (moment - origin).total_seconds() / 3600.0


def clamp_rate(rate):
    """Clamp a percent into the range the scorer's logit can take."""
    if rate < LOGIT_LO:
        return LOGIT_LO
    if rate > LOGIT_HI:
        return LOGIT_HI
    return rate


def logit_of(rate):
    """logit of a percent: ln(p / (1 - p)) with p = rate / 100."""
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def rate_from_logit(value):
    """Inverse logit back to a percent, clamped to the open interval."""
    if value > 30.0:
        return LOGIT_HI
    if value < -30.0:
        return LOGIT_LO
    p = 1.0 / (1.0 + math.exp(-value))
    out = 100.0 * p
    if out < LOGIT_LO:
        return LOGIT_LO
    if out > LOGIT_HI:
        return LOGIT_HI
    return out


def round4(value):
    """Round to 4 decimals, keeping the value a plain float."""
    return round(float(value) + 0.0, 4)


def fit_straight_line(xs, ys):
    """Least-squares y = a + b*x; returns (a, b) or None when degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mean_x
        sxx += dx * dx
        sxy += dx * (y - mean_y)
    if sxx <= 1e-12:
        return None
    slope = sxy / sxx
    return (mean_y - slope * mean_x, slope)


def input_folder():
    """Input directory: $MAKEFUN_INPUT_DIR when set, else /input."""
    chosen = os.environ.get("MAKEFUN_INPUT_DIR")
    if chosen:
        return chosen
    return "/input"


def output_folder():
    """Output directory: $MAKEFUN_OUTPUT_DIR when set, else /output."""
    chosen = os.environ.get("MAKEFUN_OUTPUT_DIR")
    if chosen:
        return chosen
    return "/output"


def list_case_paths(folder):
    """Every *.in file in the folder, in name order."""
    names = []
    try:
        entries = os.listdir(folder)
    except OSError:
        return names
    for name in entries:
        if name.endswith(".in"):
            names.append(os.path.join(folder, name))
    names.sort()
    return names


def load_case(path):
    """Load one case file as a dict."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_forecast(folder, case_id, rows):
    """Write {"trophies": [...]} for one case, creating the folder."""
    os.makedirs(folder, exist_ok=True)
    payload = {"trophies": rows}
    target = os.path.join(folder, case_id + ".out")
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
        handle.write("\n")

# ---------------------------------------------------------------------------
# Trophy archetypes and their priors.
#
# A trophy's earners rise on a saturating curve. How high that curve can go
# (its ceiling, as a share of tracked players) and how fast it gets there
# depends on what the trophy asks for. The table below is calibrated from
# the live Exophase page two days after launch (Chapter 1 already near 8%,
# the fishing challenge above 10%, late chapters near zero) and from how
# Grand Theft Auto V's trophies spread out over a long lifetime (first
# mission common, final mission and 100% completion very rare). Each row is
# (ceiling share, timescale in hours, Weibull shape, extra logit spread,
# stale-window earner growth, fresh-player earn chance). The fit to the
# case history dominates whenever the history is informative; the prior only
# steadies trophies whose history barely moves.
# ---------------------------------------------------------------------------

ARCHETYPE_PRIORS = {
    "opening":       (1.00,  12.0, 2.0, 0.05, 0.0, 1.00),
    "chapter1":      (0.85, 110.0, 2.0, 0.10, 0.075, 0.00025),
    "chapter2":      (0.65, 170.0, 2.0, 0.15, 0.05, 0.000075),
    "chapter3":      (0.50, 260.0, 2.0, 0.20, 0.0375, 0.00005),
    "chapter4":      (0.40, 330.0, 2.0, 0.25, 0.025, 0.000025),
    "finale":        (0.28, 420.0, 2.0, 0.30, 0.0125, 0.000025),
    "story_mission": (0.45, 220.0, 2.0, 0.20, 0.075, 0.0002),
    "side_mission":  (0.25, 350.0, 2.0, 0.30, 0.0375, 0.00005),
    "platinum":      (0.04, 1500.0, 2.0, 0.55, 0.0, 0.0),
    "completion":    (0.04, 1300.0, 2.0, 0.55, 0.0, 0.0),
    "grind_big":     (0.05, 700.0, 1.5, 0.40, 0.025, 0.00005),
    "grind_mid":     (0.12, 500.0, 1.5, 0.30, 0.075, 0.0005),
    "easy":          (0.35, 150.0, 1.5, 0.15, 0.125, 0.0005),
    "money":         (0.08, 450.0, 1.5, 0.35, 0.05, 0.000125),
    "combat":        (0.18, 320.0, 1.8, 0.25, 0.075, 0.00025),
    "flavour":       (0.20, 280.0, 1.8, 0.30, 0.075, 0.00025),
}


def classify_trophy(trophy):
    """Guess a trophy's archetype from its tier, name and description."""
    tier = (trophy.get("tier") or "").lower()
    text = ((trophy.get("description") or "") + " " +
            (trophy.get("name") or "")).strip()
    low = text.lower()
    if tier == "platinum":
        return "platinum"
    if "100% completion" in low:
        return "completion"
    if low.startswith("complete chapter 1."):
        return "chapter1"
    if low.startswith("complete chapter 2."):
        return "chapter2"
    if low.startswith("complete chapter 3."):
        return "chapter3"
    if low.startswith("complete chapter 4."):
        return "chapter4"
    if low.startswith("complete the final mission."):
        return "finale"
    if low.startswith("complete gone fishing."):
        return "opening"
    if low.startswith("complete the final mission for"):
        return "side_mission"
    if low.startswith("complete ") and trophy.get("secret"):
        return "story_mission"
    if ("stunts" in low or "challenges" in low or "waink" in low
            or "card sets" in low or "mission strands" in low
            or "shooting range" in low or "drone race" in low
            or "meditate at every spot" in low):
        return "grind_big"
    if ("discover 100 vehicles" in low or "win 20 different races" in low
            or "explore all of leonida" in low):
        return "grind_mid"
    if ("fishing challenge" in low or "arcade game" in low
            or "try each type of activity" in low):
        return "easy"
    if ("laundering" in low or "spend $100,000" in low
            or "garage properties" in low or "register 10 vehicles" in low):
        return "money"
    if ("combat focus" in low or "jewelry store" in low
            or "bank vaults" in low or "armored truck" in low
            or "illegal operation" in low or "rob 3 shops" in low
            or "wanted level" in low or "vcia" in low or "rhinos" in low):
        return "combat"
    return "flavour"


def prior_for(trophy):
    """The prior row for a trophy: (ceiling, tau, shape, spread, stale, fresh)."""
    return ARCHETYPE_PRIORS[classify_trophy(trophy)]

def distinct_snapshots(case):
    """Compress the history to reads where something actually changed.

    Exophase refreshes discretely, so long runs of reads repeat the same
    page. Keeping every repeat would overweight the stale stretch in every
    fit, so only the first read of each distinct page is kept. Returns a
    list of (hours since release, players tracked, rates list).
    """
    seen = []
    previous = None
    for read in case.get("history") or []:
        players = read.get("players_tracked")
        rates = read.get("rates") or []
        key = (players, tuple(rates))
        if key == previous:
            continue
        previous = key
        try:
            moment = parse_moment(read.get("at") or read.get("slot"))
        except (ValueError, TypeError):
            continue
        seen.append((hours_since_release(moment), players, list(rates)))
    return seen


def weibull_rise(t, tau, shape):
    """Saturating rise 1 - exp(-(t / tau) ** shape), t in hours."""
    if t <= 0.0:
        return 0.0
    scaled = t / tau
    if scaled > 12.0:
        return 1.0
    return 1.0 - math.exp(-(scaled ** shape))


def _log_grid(lo, hi, count):
    """`count` points spaced evenly in log space from lo to hi."""
    start = math.log(lo)
    step = (math.log(hi) - start) / max(count - 1, 1)
    return [math.exp(start + step * i) for i in range(count)]


def fit_players(snapshots, target_u):
    """Fit players tracked as a saturating curve and extrapolate.

    P(u) = P_inf * (1 - exp(-u / tau)), u measured in hours since the
    first read, fitted on log P over a grid of tau. Returns
    (P at target, P_inf, tau, log-residual sd, number of points) or None
    when the history carries no usable growth signal.
    """
    if len(snapshots) < 2:
        return None
    base = snapshots[0][0]
    us = []
    logs = []
    for t, players, _rates in snapshots:
        if players and players > 0:
            us.append(max(t - base, 0.0))
            logs.append(math.log(players))
    if len(us) < 2 or max(us) <= 0.0:
        return None
    best = None
    for tau in _log_grid(6.0, 3000.0, 41):
        shifted = []
        usable = True
        for u, lp in zip(us, logs):
            rise = 1.0 - math.exp(-u / tau) if u > 0 else 0.0
            if rise <= 1e-6:
                continue
            shifted.append(lp - math.log(rise))
        if len(shifted) < 2:
            continue
        log_inf = sum(shifted) / len(shifted)
        sse = 0.0
        used = 0
        for u, lp in zip(us, logs):
            rise = 1.0 - math.exp(-u / tau) if u > 0 else 0.0
            if rise <= 1e-6:
                continue
            diff = lp - (log_inf + math.log(rise))
            sse += diff * diff
            used += 1
        if used < 2:
            continue
        score = sse / used
        if best is None or score < best[0]:
            best = (score, log_inf, tau, used)
    if best is None:
        return None
    score, log_inf, tau, used = best
    p_inf = math.exp(log_inf)
    u_target = max(target_u - base, 0.0)
    p_target = p_inf * (1.0 - math.exp(-u_target / tau)) if u_target > 0 else p_inf
    resid = math.sqrt(score) if score > 0 else 0.0
    return (p_target, p_inf, tau, resid, used)


def fit_earners_grid(times, earners):
    """Grid of Weibull fits for one trophy's earners.

    E(t) = A * (1 - exp(-(t / tau) ** k)), fitted on log(E + 0.5) over a
    grid of tau and k, with A by least squares for each pair. Returns a
    list of (tau, shape, log A, sse, used). The caller averages over this
    grid with a prior, because on early data A and tau are barely
    separable: many pairs fit the history equally well and only the prior
    tells them apart at the target.
    """
    points = []
    for t, e in zip(times, earners):
        if e is None:
            continue
        points.append((max(t, 0.25), max(e, 0.0)))
    if len(points) < 2:
        return []
    xs = [p[0] for p in points]
    ys = [math.log(p[1] + 0.5) for p in points]
    grid = []
    for tau in _log_grid(12.0, 2500.0, 33):
        for shape in (1.0, 1.5, 2.0, 2.5, 3.0):
            rises = [weibull_rise(x, tau, shape) for x in xs]
            terms = []
            for rise, y in zip(rises, ys):
                if rise <= 0.01:
                    continue
                terms.append(y - math.log(rise))
            if len(terms) < 2:
                continue
            log_a = sum(terms) / len(terms)
            sse = 0.0
            for rise, y in zip(rises, ys):
                if rise <= 0.01:
                    continue
                diff = y - (log_a + math.log(rise))
                sse += diff * diff
            grid.append((tau, shape, log_a, sse, len(terms)))
    return grid


def average_earners(grid, target_t, a_prior, tau_prior, p_inf, ceiling):
    """Posterior mean of earners at the target over the Weibull grid.

    Weights are likelihood times prior: the likelihood from the fit's
    squared log error with a noise floor (the curve family never matches
    reality exactly, so a near-perfect in-sample fit must not collapse
    the average onto one point), the prior lognormal around the
    archetype's timescale and ceiling. Returns (mean earners, relative
    sd of earners) or None when the grid is empty.
    """
    if not grid:
        return None
    best_sse = min(g[3] for g in grid)
    best_used = max(g[4] for g in grid)
    var = best_sse / max(best_used, 1)
    if var < 0.04:
        var = 0.04
    log_tau_prior = math.log(tau_prior)
    log_a_prior = math.log(a_prior)
    cap = ceiling * p_inf * 1.5
    total = 0.0
    mean = 0.0
    sq = 0.0
    for tau_g, k_g, log_a_g, sse_g, _used in grid:
        a_g = math.exp(log_a_g)
        if a_g > cap:
            a_g = cap
            log_a_g = math.log(a_g)
        like = math.exp(-(sse_g - best_sse) / (2.0 * var))
        prior_tau = math.exp(-0.5 * ((math.log(tau_g) - log_tau_prior) / 0.7) ** 2)
        prior_a = math.exp(-0.5 * ((log_a_g - log_a_prior) / 0.7) ** 2)
        weight = like * prior_tau * prior_a
        e_g = a_g * weibull_rise(target_t, tau_g, k_g)
        total += weight
        mean += weight * e_g
        sq += weight * e_g * e_g
    if total <= 0.0:
        return None
    mean /= total
    var_e = sq / total - mean * mean
    if var_e < 0.0:
        var_e = 0.0
    rel = math.sqrt(var_e) / mean if mean > 0 else 1.0
    return (mean, rel)


def trend_median(times, rates, target_t):
    """House-trend-style median: logit(rate) linear in ln(1 + hours).

    Uses the last 24 hours of reads. Returns (median rate, residual sd)
    or None when fewer than 3 usable reads exist.
    """
    if not times:
        return None
    latest = max(times)
    xs = []
    ys = []
    for t, r in zip(times, rates):
        if r is None or t < latest - 24.0:
            continue
        xs.append(math.log1p(max(t, 0.0)))
        ys.append(logit_of(r))
    if len(xs) < 3:
        return None
    fit = fit_straight_line(xs, ys)
    if fit is None:
        return None
    intercept, slope = fit
    if slope < 0.0:
        slope = 0.0
    resid = 0.0
    for x, y in zip(xs, ys):
        diff = y - (intercept + slope * x)
        resid += diff * diff
    resid = math.sqrt(resid / len(xs))
    median = rate_from_logit(intercept + slope * math.log1p(max(target_t, 0.0)))
    return (median, resid)

def _quantiles_from(median, spread):
    """Five quantiles around a median rate on the logit scale."""
    if median >= 100.0:
        return [100.0, 100.0, 100.0, 100.0, 100.0]
    centre = logit_of(median)
    out = []
    for z in Z_VALUES:
        value = rate_from_logit(centre + z * spread)
        out.append(round4(value))
    for i in range(1, len(out)):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _stale_median(trophy, last_rate, last_players):
    """Forecast when the history is one frozen page (launch night).

    Exophase has not refreshed for hours, so the next refresh jumps the
    tracked players (about sixfold on launch night, 638 to 3,868) while
    earners only grew a little. The rate therefore drops for every trophy
    but the opening one. Newly tracked players add a few earners, at a
    rate set by the trophy's archetype.
    """
    prior = prior_for(trophy)
    _ceiling, _tau, _shape, _spread, stale_growth, fresh_prob = prior
    if last_rate is None:
        last_rate = 0.0
    if last_rate >= 100.0:
        return 100.0
    players_new = last_players * 6.0
    earners_old = last_rate * last_players / 100.0
    earners_new = (earners_old * (1.0 + stale_growth)
                   + (players_new - last_players) * fresh_prob)
    rate = 100.0 * earners_new / players_new if players_new > 0 else last_rate
    if rate < 0.01:
        rate = 0.01
    if rate > 99.99:
        rate = 99.99
    return rate


def _forecast_one(trophy, series, target_t, horizon_days, players_fit):
    """Forecast one trophy: returns (median rate, logit spread)."""
    prior = prior_for(trophy)
    ceiling, tau_prior, k_prior, spread_extra = prior[0], prior[1], prior[2], prior[3]
    times = [point[0] for point in series]
    rates = [point[1] for point in series]
    last_rate = rates[-1] if rates else None
    if last_rate is not None and last_rate >= 100.0:
        return (100.0, 0.05)
    if not series:
        return (max(0.05, 100.0 * ceiling * 0.02), 1.2 + spread_extra)

    players_target = None
    if players_fit is not None:
        players_target = players_fit[0]
    if players_target is None or players_target <= 0:
        players_target = series[-1][2] if series else 1.0

    earners = []
    for _t, r, players in series:
        if r is None or players is None or players <= 0:
            earners.append(None)
        else:
            earners.append(r * players / 100.0)
    grid = fit_earners_grid(times, earners)
    median_ep = None
    resid = 0.0
    if grid:
        p_inf = players_fit[1] if players_fit is not None else players_target
        if p_inf <= 0:
            p_inf = players_target
        a_prior = ceiling * p_inf
        if a_prior <= 0:
            a_prior = 0.5
        averaged = average_earners(grid, target_t, a_prior, tau_prior,
                                   p_inf, ceiling)
        if averaged is not None:
            e_mean, e_rel = averaged
            resid = e_rel
            median_ep = 100.0 * e_mean / players_target
    if median_ep is None or median_ep <= 0:
        median_ep = max(0.02, 100.0 * ceiling * 0.02)
    if last_rate is not None and last_rate <= 0.0 and median_ep <= 0.0:
        median_ep = 0.02

    trend = trend_median(times, rates, target_t)
    median_logit = logit_of(median_ep)
    disagreement = 0.0
    if trend is not None:
        trend_rate, trend_resid = trend
        median_logit = 0.80 * median_logit + 0.20 * logit_of(trend_rate)
        disagreement = abs(logit_of(median_ep) - logit_of(trend_rate))
        resid = max(resid, trend_resid * 0.5)
    median = rate_from_logit(median_logit)

    spread = 0.20 * math.sqrt(max(horizon_days, 0.01))
    spread += spread_extra
    spread += min(0.5 * resid, 0.25)
    spread += min(0.30 * disagreement, 0.30)
    if spread < 0.15:
        spread = 0.15
    if spread > 1.8:
        spread = 1.8
    # Backtests on synthetic finals put the best spread near half of the
    # textbook width: the fitted curves already carry most of the
    # information, so a wide band mostly pays for itself. Keep 0.7 of the
    # computed width as insurance against the real data differing from
    # the synthetic finals.
    spread = spread * 0.7
    if spread < 0.15:
        spread = 0.15
    return (median, spread)


def forecast_case(case):
    """Forecast every trophy of one case; returns the output rows."""
    trophies = case.get("trophies") or []
    horizon = case.get("horizon_minutes") or 0
    horizon_days = horizon / 1440.0
    try:
        target_moment = parse_moment(case.get("target"))
        target_t = hours_since_release(target_moment)
    except (ValueError, TypeError):
        target_t = 168.0
    try:
        checkpoint = parse_moment(case.get("at"))
        checkpoint_t = hours_since_release(checkpoint)
    except (ValueError, TypeError):
        checkpoint_t = target_t

    snapshots = distinct_snapshots(case)
    stale = len(snapshots) <= 1
    players_fit = None
    if not stale:
        players_fit = fit_players(snapshots, checkpoint_t)
        if players_fit is not None:
            target_u = target_t
            base = snapshots[0][0]
            u_target = max(target_u - base, 0.0)
            p_inf = players_fit[1]
            tau_p = players_fit[2]
            p_target = p_inf * (1.0 - math.exp(-u_target / tau_p)) if u_target > 0 else p_inf
            players_fit = (p_target, p_inf, tau_p, players_fit[3], players_fit[4])

    rows = []
    for index, trophy in enumerate(trophies):
        trophy_id = trophy.get("id", index + 1)
        try:
            series = []
            for t, players, rates in snapshots:
                rate = rates[index] if index < len(rates) else None
                series.append((t, rate, players))
            last_rate = series[-1][1] if series else None
            last_players = series[-1][2] if series else 1
            if stale:
                median = _stale_median(trophy, last_rate, last_players or 1)
                spread = 0.30 * math.sqrt(max(horizon_days, 0.01)) + 0.70
                spread += prior_for(trophy)[3]
                if spread > 1.8:
                    spread = 1.8
                rows.append({"id": trophy_id,
                             "q": _quantiles_from(median, spread)})
            else:
                median, spread = _forecast_one(
                    trophy, series, target_t, horizon_days, players_fit)
                rows.append({"id": trophy_id,
                             "q": _quantiles_from(median, spread)})
        except Exception:
            fallback = 0.5
            try:
                fallback = series[-1][1]
            except (NameError, IndexError, TypeError):
                fallback = 0.5
            if fallback is None:
                fallback = 0.5
            rows.append({"id": trophy_id,
                         "q": _quantiles_from(fallback, 1.0)})
    rows.sort(key=lambda row: row["id"])
    return rows


def main():
    """Run every case in the input folder and write its forecast."""
    in_dir = input_folder()
    out_dir = output_folder()
    for path in list_case_paths(in_dir):
        try:
            case = load_case(path)
        except (OSError, ValueError):
            continue
        case_id = case.get("id")
        if not case_id:
            case_id = os.path.splitext(os.path.basename(path))[0]
        try:
            rows = forecast_case(case)
        except Exception:
            rows = []
            for index, trophy in enumerate(case.get("trophies") or []):
                rows.append({"id": trophy.get("id", index + 1),
                             "q": [0.01, 0.05, 0.1, 0.5, 2.0]})
        write_forecast(out_dir, case_id, rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
