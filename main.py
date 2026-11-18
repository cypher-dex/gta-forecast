#!/usr/bin/env python3
"""Launch-day Twitch viewer forecast for Grand Theft Auto VI (make.fun gta6-twitch-ccu-forecast-01).

Analog ensemble over past launches.  Every launch in the case's `comparables`
gives a path of log ratios ln(v_h / v_1) over its own release day (hour 1 =
00:00-01:00 UTC).  Each forecast hour is a weighted Gaussian-kernel mixture of
those ratios.  A comparable's weight is

    prior(unlock style, platforms, tier) x kernel(features seen at the checkpoint)

where the features are the hour-1 size ln(v_1) and the hour-1 print against the
highest hour of the 12 before it (has the launch surge already peaked?).  Big
hour-1 prints that already sit below the pre-midnight peak stay flat for the
day; small ones keep climbing.  Each ratio is also shifted by
BETA x (size gap) so a target larger than every comparable is extrapolated.

    python3 main.py              forecast every *.in in $MAKEFUN_INPUT_DIR
    python3 main.py --loo DIR    leave-one-out backtest over the comparables in DIR
"""
import json
import math
import os
import re
import sys
from datetime import datetime

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
HOURS = range(2, 25)

# Tuned by leave-one-out over the comparables (see --loo); chosen on a flat plateau.
PARAMS = {
    "pc_weight": 0.35,        # launches that include PC, often on their own unlock clock
    "global_weight": 0.15,    # single worldwide unlock time instead of rolling midnight
    "early_weight": 0.5,      # early-access contaminated release day
    "weak_weight": 0.2,       # free-to-play / slow-burn launches
    "small_weight": 0.2,      # hour 1 below `small_v1` viewers: launch not yet started
    "small_v1": 2000.0,
    "size_bw": 1.0,           # kernel bandwidth on ln(v_1)
    "peak_bw": 0.5,           # kernel bandwidth on ln(v_1 / max of hours -11..0)
    "beta": -0.1,             # log-ratio shift per unit of ln(v_1) gap (bigger -> flatter)
    "bw0": 0.10,              # mixture bandwidth at hour 2 (log units)
    "bw1": 0.06,              # bandwidth growth per sqrt(hour - 1)
}


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hour_of(day_start, at):
    """Release-day hour number (1 = 00:00-01:00 UTC) holding instant `at`."""
    return int((at - day_start).total_seconds() // 3600) + 1


def hourly_map(day_start, rows):
    ds = parse_ts(day_start)
    out = {}
    for r in rows or []:
        v = r.get("viewers")
        if v is not None:
            out[hour_of(ds, parse_ts(r["from"]))] = v
    return out


def features(series, v1):
    """(ln v_1, ln(v_1 / max of hours -11..0) or None)."""
    pre = [series[k] for k in range(-11, 1) if k in series]
    peak = math.log(max(v1, 1) / max(max(pre), 1)) if pre else None
    return math.log(max(v1, 1)), peak


class Analog:
    __slots__ = ("name", "v1", "ratios", "prior", "feat", "console", "glob")

    def __init__(self, comp, prm):
        s = hourly_map(comp["day_start"], comp.get("hourly"))
        self.name = comp["name"]
        self.v1 = s.get(1) or 0
        self.ratios = {}
        if self.v1 > 0:
            self.ratios = {h: math.log(max(s[h], 1) / self.v1) for h in HOURS if h in s}
        unlock = comp.get("unlock", "").lower()
        plat = comp.get("platforms", "").lower()
        tier = re.search(r"tier:\s*([a-z-]+)", unlock)
        tier = tier.group(1) if tier else ""
        self.console = "pc" not in plat
        self.glob = bool(re.search(
            r"single (simultaneous )?global|simultaneous global|simultaneous across|single time", unlock))
        w = 0.0 if tier == "target-history" or not self.ratios else 1.0
        if not self.console:
            w *= prm["pc_weight"]
        if self.glob:
            w *= prm["global_weight"]
        if tier == "early-access-contaminated":
            w *= prm["early_weight"]
        elif tier == "weak":
            w *= prm["weak_weight"]
        if self.v1 < prm["small_v1"]:
            w *= prm["small_weight"]
        self.prior = w
        self.feat = features(s, self.v1) if self.v1 > 0 else (0.0, None)


def kernel(a, b, prm):
    k = math.exp(-0.5 * ((a[0] - b[0]) / prm["size_bw"]) ** 2)
    if a[1] is not None and b[1] is not None:
        k *= math.exp(-0.5 * ((a[1] - b[1]) / prm["peak_bw"]) ** 2)
    return k


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mixture_quantile(points, bw, tau):
    """Quantile of sum_i w_i N(x_i, bw^2); points = [(x, w)] with sum w = 1."""
    lo = min(x for x, _ in points) - 8 * bw
    hi = max(x for x, _ in points) + 8 * bw
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if sum(w * norm_cdf((mid - x) / bw) for x, w in points) < tau:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def forecast_logs(feat, analogs, prm):
    """{h: [5 log ratios to hour 1]} for a target with features `feat`."""
    pool = []
    for a in analogs:
        w = a.prior * kernel(feat, a.feat, prm) if a.prior > 0 else 0.0
        if w > 1e-12:
            pool.append((w, prm["beta"] * (feat[0] - a.feat[0]), a.ratios))
    out = {}
    for h in HOURS:
        bw = prm["bw0"] + prm["bw1"] * math.sqrt(h - 1)
        pts = [(r[h] + shift, w) for w, shift, r in pool if h in r]
        tot = sum(w for _, w in pts)
        if not pts or tot <= 0:
            out[h] = [z * (0.15 * math.sqrt(h - 1)) for z in Z]
            continue
        pts = [(x, w / tot) for x, w in pts]
        out[h] = [mixture_quantile(pts, bw, t) for t in LEVELS]
    return out


def format_rows(p1, logs):
    rows = []
    for h in HOURS:
        q, prev = [], 0.01
        for lr in logs[h]:
            v = min(max(round(p1 * math.exp(lr), 2), prev, 0.01), 1e9)
            q.append(v)
            prev = v
        rows.append({"hour": h, "q": q})
    return rows


def forecast_case(case, prm=PARAMS):
    p1 = max(float(case["known"][0]["viewers"]), 1.0)
    series = hourly_map(case["day_start"], case.get("history_hourly"))
    series[1] = p1
    analogs = [Analog(c, prm) for c in case.get("comparables") or []]
    return {"hours": format_rows(p1, forecast_logs(features(series, p1), analogs, prm))}


def pinball(y, qs):
    """Mean over the five levels of the contest's log-scale pinball loss."""
    ly = math.log(max(y, 1))
    tot = 0.0
    for tau, q in zip(LEVELS, qs):
        d = ly - math.log(q)
        tot += 2 * max(tau * d, (tau - 1) * d)
    return tot / len(LEVELS)


def baseline_q(p1, h):
    return [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z]


def loo(indir, prm=PARAMS, verbose=True):
    comps = {}
    for name in sorted(os.listdir(indir)):
        if name.endswith(".in"):
            with open(os.path.join(indir, name), encoding="utf-8") as f:
                for c in json.load(f).get("comparables") or []:
                    comps.setdefault(c["name"], c)
    analogs = [Analog(c, prm) for c in comps.values()]
    res = []
    for t in analogs:
        if t.prior <= 0:
            continue
        rest = [a for a in analogs if a is not t]
        rows = {r["hour"]: r["q"] for r in format_rows(t.v1, forecast_logs(t.feat, rest, prm))}
        ys = {h: t.v1 * math.exp(lr) for h, lr in t.ratios.items()}
        ours = sum(pinball(y, rows[h]) for h, y in ys.items()) / len(ys)
        base = sum(pinball(y, baseline_q(t.v1, h)) for h, y in ys.items()) / len(ys)
        res.append((t, ours, base))
    if verbose:
        for t, a, b in res:
            print(f"{t.name[:30]:30s} {'console' if t.console else 'pc     '} ours {a:.3f} base {b:.3f}")
    subsets = (
        ("all", res),
        ("console-only", [r for r in res if r[0].console]),
        ("target-like", [r for r in res if not r[0].glob and r[0].v1 > 5000]),
    )
    for label, sel in subsets:
        if sel:
            print(f"{label:13s} n={len(sel):2d} ours {sum(r[1] for r in sel) / len(sel):.4f}"
                  f" base {sum(r[2] for r in sel) / len(sel):.4f}")
    return res


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--loo":
        loo(sys.argv[2])
        return
    indir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    outdir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(outdir, exist_ok=True)
    for name in sorted(os.listdir(indir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(indir, name), encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case)
        with open(os.path.join(outdir, name[:-3] + ".out"), "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
