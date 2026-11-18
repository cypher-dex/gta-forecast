"""Launch-day Twitch viewer forecast: a kernel-weighted analog ensemble.

For every past launch in the case's comparables, the log ratio of each hour of
its release day to its hour 1 is one analog path. Analogs are weighted by how
much their run-up into hour 1 looks like the case's own (growth over the last
hour, over the last six hours, against the previous day's peak and against the
quiet weeks before) and by a prior on the kind of launch (rolling midnight
unlock, console-only, early access). Each analog contributes a normal kernel in
log space; the five quantiles of each hour are read off the weighted mixture.
When the case carries minute reads inside hour 1, the trend over its last
minutes nudges the first hours of the forecast.
"""

import glob
import json
import math
import os
from datetime import datetime

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
GTA_TWITCH_ID = "1867279146"

# Tuned by leave-one-out over the comparables (rolling/console launches).
BANDWIDTH = 1.0
FEATURE_SCALE = (1.0, 1.0, 1.0, 0.2)
PRIOR_POWER = 2.0
KERNEL_SD_BASE = 0.10
KERNEL_SD_SLOPE = 0.03
# Minute-trend nudge: share of the hour-1 tail trend carried into hour 2,
# decaying by this factor each further hour.
TREND_CARRY = 0.6
TREND_DECAY = 0.6


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_index(start, at):
    return int((parse_time(at) - start).total_seconds() // 3600)


def safe_log(x):
    return math.log(max(float(x), 1.0))


def hourly_map(rows, start, key="from"):
    """Hour offset from `start` -> viewers, for rows with a viewers count."""
    out = {}
    for row in rows or []:
        v = row.get("viewers")
        if v is None or row.get(key) is None:
            continue
        out[hour_index(start, row[key])] = float(v)
    return out


def features(series):
    """Run-up features of a launch, from its hourly map relative to day start."""
    v1 = series.get(0)
    lv = safe_log(v1)
    last6 = [series[k] for k in range(-6, 0) if k in series]
    last24 = [series[k] for k in range(-24, 0) if k in series]
    quiet = sorted(series[k] for k in range(-14 * 24, -24) if k in series)
    growth_1h = lv - safe_log(last6[-1]) if last6 else 0.0
    growth_6h = lv - safe_log(sum(last6) / len(last6)) if last6 else 0.0
    vs_peak = lv - safe_log(max(last24)) if last24 else 0.0
    if quiet:
        mid = len(quiet) // 2
        median = quiet[mid] if len(quiet) % 2 else (quiet[mid - 1] + quiet[mid]) / 2
        vs_quiet = lv - safe_log(median)
    else:
        vs_quiet = 3.0
    return (growth_1h, growth_6h, vs_peak, vs_quiet)


def launch_prior(comparable):
    unlock = (comparable.get("unlock") or "").lower()
    platforms = (comparable.get("platforms") or "").lower()
    rolling = "midnight local" in unlock
    early = ("early access" in unlock and "no early access" not in unlock) or "early-access" in unlock
    simultaneous = "simultaneous" in unlock or "single global" in unlock
    console_only = "pc" not in platforms
    w = 1.0
    if early:
        w *= 0.4
    if "tier: weak" in unlock:
        w *= 0.3
    if rolling:
        w *= 2.0
    if console_only:
        w *= 1.5
    if simultaneous and not rolling:
        w *= 0.5
    return w


def analogs_of(case):
    out = []
    for comp in case.get("comparables") or []:
        start = parse_time(comp["day_start"])
        series = hourly_map(comp.get("hourly"), start)
        if series.get(0) is None:
            continue  # no hour 1 on its day (e.g. the target's own pre-release history)
        base = safe_log(series[0])
        ratios = {h: safe_log(series[h - 1]) - base for h in range(2, 25) if (h - 1) in series}
        if not ratios:
            continue
        out.append((features(series), ratios, launch_prior(comp)))
    return out


def target_series(case):
    start = parse_time(case["day_start"])
    series = hourly_map(case.get("history_hourly"), start)
    if not series:
        # Fall back to bucketing the raw points into hours.
        buckets = {}
        for p in case.get("history") or []:
            if p.get("viewers") is None:
                continue
            buckets.setdefault(hour_index(start, p["at"]), []).append(float(p["viewers"]))
        series = {k: sum(v) / len(v) for k, v in buckets.items()}
    series[0] = float(case["known"][0]["viewers"])
    # The GTA VI category's own long history sits among the comparables: use it
    # for the quiet weeks the recorder's short history does not reach.
    gid = str((case.get("game") or {}).get("twitch_id") or "")
    for comp in case.get("comparables") or []:
        name = (comp.get("name") or "").lower()
        if gid == GTA_TWITCH_ID and name.startswith("grand theft auto vi") and "before" in name:
            for k, v in hourly_map(comp.get("hourly"), start).items():
                if k < 0 and k not in series:
                    series[k] = v
    return series


def minute_trend(case):
    """Log of the last 15 minutes' mean over the hour-1 print, when minute reads exist."""
    start = parse_time(case["day_start"])
    checkpoint = parse_time(case["checkpoint"])
    tail = []
    for p in case.get("history") or []:
        if p.get("viewers") is None:
            continue
        t = parse_time(p["at"])
        if start <= t < checkpoint and (checkpoint - t).total_seconds() <= 15 * 60:
            tail.append(float(p["viewers"]))
    if len(tail) < 5:
        return 0.0
    trend = safe_log(sum(tail) / len(tail)) - safe_log(case["known"][0]["viewers"])
    return max(-0.5, min(0.5, trend))


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantiles(means, sds, weights, levels):
    total = sum(weights)
    lo0 = min(m - 6 * s for m, s in zip(means, sds))
    hi0 = max(m + 6 * s for m, s in zip(means, sds))
    out = []
    for tau in levels:
        lo, hi = lo0, hi0
        for _ in range(40):
            mid = (lo + hi) / 2
            c = sum(w * normal_cdf((mid - m) / s) for m, s, w in zip(means, sds, weights)) / total
            if c < tau:
                lo = mid
            else:
                hi = mid
        out.append((lo + hi) / 2)
    return out


def baseline_logs(h):
    z = [-1.6448536, -0.6744898, 0.0, 0.6744898, 1.6448536]
    return [zz * 0.15 * math.sqrt(h - 1) for zz in z]


def forecast(case):
    levels = case.get("levels") or LEVELS
    print_1 = max(float(case["known"][0]["viewers"]), 1.0)
    target = features(target_series(case))
    analogs = analogs_of(case)

    weights = []
    for feats, _, prior in analogs:
        d2 = sum(((a - b) * s) ** 2 for a, b, s in zip(feats, target, FEATURE_SCALE))
        weights.append(prior ** PRIOR_POWER * math.exp(-d2 / (2 * BANDWIDTH ** 2)))
    if sum(weights) <= 1e-300:
        weights = [prior ** PRIOR_POWER for _, _, prior in analogs]

    trend = minute_trend(case)
    hours = [row["hour"] for row in case.get("forecast") or []] or list(range(2, 25))
    rows = []
    for h in hours:
        means, sds, ws = [], [], []
        for (_, ratios, _), w in zip(analogs, weights):
            if h in ratios and w > 0:
                means.append(ratios[h])
                sds.append(KERNEL_SD_BASE + KERNEL_SD_SLOPE * math.sqrt(h - 1))
                ws.append(w)
        logs = mixture_quantiles(means, sds, ws, levels) if ws else baseline_logs(h)
        shift = trend * TREND_CARRY * TREND_DECAY ** (h - 2)
        q = []
        for x in logs:
            value = print_1 * math.exp(x + shift)
            value = round(min(max(value, 0.01), 1e9), 2)
            if q and value < q[-1]:
                value = q[-1]
            q.append(value)
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        with open(path, encoding="utf-8") as f:
            case = json.load(f)
        case_id = case.get("id") or os.path.basename(path)[:-3]
        try:
            result = forecast(case)
        except Exception:
            p = max(float(case["known"][0]["viewers"]), 1.0)
            result = {"hours": [
                {"hour": h, "q": [round(p * math.exp(x), 2) for x in baseline_logs(h)]}
                for h in range(2, 25)
            ]}
        with open(os.path.join(out_dir, case_id + ".out"), "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
