"""Fuel tank leak detection: a simulator, three detectors, and the regulator's test.

Every gallon is accounted for in a tank: what was there, plus deliveries,
minus sales, minus anything lost. Statistical inventory reconciliation (SIR)
finds a leak as the part of that balance nothing else explains. The federal
standard (40 CFR 280.43(h)) asks a precise question of any method: detect
0.2 gal/h with probability at least 0.95, at a false-alarm probability no
more than 0.05, with a threshold no more than half the detectable rate, and
report the rate itself.

This module asks that question of three methods, on simulated tanks where the
truth is known:

    classical SIR      variance regressed on time, cumulative sales and
                       cumulative deliveries — the certified incumbent
    plain network      learns the hourly change from a previous leak-free
                       month, then reads the test month's residual
    conserving         the same mass balance with the leak as its own term,
                       plus a small network for the tank chart's
                       level-dependent error — the one thing SIR cannot
                       model. The network only reinterprets what the gauge
                       reads; it has nowhere to put a lost gallon.

The simulation is mine, so the result is a statement about these
assumptions: a 10,000-gallon tank selling about 20,000 gallons a month (the
median throughput in one certification listing), thermal expansion at 0.00069
per °F, dispenser meters off by up to ±0.3%, delivery receipts off by ±0.3%,
gauge noise, a temperature probe that reads the average imperfectly, and a
tank chart wrong by up to about 0.3% depending on the level. Real tanks have
more ways to be wrong than this, and the only test that settles anything is
real gauge data.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np

BETA = 0.00069          # gasoline, volume per volume per °F
CAPACITY = 10_000.0     # gallons
HOURS = 720             # a 30-day monitoring period
THROUGHPUT = 20_000.0   # gallons a month: a moderately busy station, so a
                        # 10,000-gallon tank takes several deliveries


def tank(rng: np.random.Generator) -> Dict[str, float]:
    """What stays the same about one tank from month to month: its dispenser
    meters and the error in its chart."""
    return {"meter": float(rng.uniform(-0.003, 0.003)),
            "phase": float(rng.uniform(0, 2 * math.pi)),
            "wobble": float(rng.uniform(0.5, 1.0))}


def simulate(leak_gph: float, rng: np.random.Generator,
             chart_error: float = 0.003,
             fixed: Dict[str, float] = None,
             night: float = 0.08) -> Dict[str, np.ndarray]:
    """One tank-month, hourly, with everything a station actually records."""
    hours = np.arange(HOURS)
    hour_of_day = hours % 24
    # trading hours carry the sales; nights are almost quiet, which is what
    # lets a leak (constant) be told apart from a meter error (per gallon sold)
    shape = np.where((hour_of_day >= 6) & (hour_of_day <= 22),
                     1.0 + 0.6 * np.sin((hour_of_day - 6) / 16 * math.pi), night)
    shape = shape / shape.mean()
    # Days differ — weekends, weather, a busy Friday. That variation is what
    # separates a meter error (proportional to gallons sold) from a leak
    # (proportional to hours passed). Measured: with every day alike,
    # cumulative sales grow almost exactly linearly in time, the two are
    # collinear, and classical SIR estimated -7.6 gal/h for a 0.2 leak.
    day = hours // 24
    weekday = np.array([0.85, 0.9, 0.95, 1.0, 1.25, 1.2, 0.85])[day % 7]
    daily = weekday * rng.lognormal(0, 0.2, HOURS // 24 + 1)[day]
    sales = np.maximum(0.0, (THROUGHPUT / HOURS) * shape * daily
                       * rng.lognormal(0, 0.3, HOURS))

    fixed = fixed or tank(rng)
    meter = fixed["meter"]                      # dispensers read high or low
    receipt_error = 0.003                       # bill of lading versus truth
    # a smooth, level-dependent chart error: the gauge's idea of the shape of
    # the tank is wrong by a different amount at different heights
    phase, wobble = fixed["phase"], fixed["wobble"]

    def chart(volume):
        level = volume / CAPACITY
        return chart_error * CAPACITY * wobble * np.sin(2.4 * math.pi * level + phase) / 2.4

    temp = (60 + 8 * math.sin(rng.uniform(0, 2 * math.pi))
            + 2.0 * np.sin(2 * math.pi * hours / 24) + rng.normal(0, 0.2, HOURS))

    net = np.empty(HOURS)
    delivered = np.zeros(HOURS)
    reported_delivery = np.zeros(HOURS)
    volume = rng.uniform(4000, 8500)
    for t in range(HOURS):
        if volume < 3500 and 6 <= hour_of_day[t] <= 18:
            drop = rng.uniform(5000, 7000)
            delivered[t] = drop
            reported_delivery[t] = drop * (1 + rng.normal(0, receipt_error))
            volume += drop
        volume -= sales[t] * (1 + meter) + leak_gph
        net[t] = volume

    gross = net * (1 + BETA * (temp - 60))
    gauge = gross + chart(net) + rng.normal(0, 0.8, HOURS)
    probe = temp + rng.normal(0, 0.15, HOURS)   # one probe, a stratified tank
    return {"gauge": gauge, "temp": probe, "sales": sales,
            "delivery": reported_delivery, "hour": hour_of_day.astype(float),
            "leak": leak_gph, "meter": meter, "tank": fixed}


def _net(month):
    """What the gauge says, compensated to 60 °F as every SIR method does."""
    return month["gauge"] / (1 + BETA * (month["temp"] - 60))


def _variance(month):
    measured = _net(month)
    # The first reading already includes hour 0's movements, so the book
    # starts there and adds hours 1..t. Lagging it one hour — as the first
    # version did — left the book a truckload short for the hour of every
    # delivery: 6,000-gallon spikes that made every fit meaningless.
    flow = month["delivery"] - month["sales"]
    book = measured[0] + np.concatenate([[0.0], np.cumsum(flow[1:])])
    return book - measured, measured


# --------------------------------------------------------------------------
# the three detectors; each returns an estimated leak rate in gal/h
# --------------------------------------------------------------------------

def classical_sir(month) -> float:
    """Variance regressed on time, cumulative sales and cumulative deliveries.

    The slope on time is the leak; the slope on sales absorbs a meter error;
    the slope on deliveries absorbs a receipt bias. This is the shape of the
    certified methods, without any vendor's refinements."""
    variance, _ = _variance(month)
    t = np.arange(HOURS, dtype=float)
    X = np.column_stack([np.ones(HOURS), t, np.cumsum(month["sales"]),
                         np.cumsum(month["delivery"])])
    coef, *_ = np.linalg.lstsq(X, variance, rcond=None)
    return float(coef[1])


def meter_error(month) -> float:
    """The meter error, from the hourly balance's slope on hourly sales —
    pinned down by the day, when sales swing widely."""
    variance, _ = _variance(month)
    change = np.diff(variance)
    sold = month["sales"][1:]
    keep = (month["delivery"][1:] == 0) & (month["delivery"][:-1] == 0)
    X = np.column_stack([np.ones(keep.sum()), sold[keep]])
    coef, *_ = np.linalg.lstsq(X, change[keep], rcond=None)
    return float(coef[1])


def quiet_hours(month, correct_meter: bool = False) -> float:
    """The balance, read only when almost nothing moves.

    Overnight there are no deliveries and only a trickle of sales, so the
    fuel that disappears beyond what was sold is the leak. Each night gives
    one estimate from the readings at the start and end of its quiet stretch
    — the gauge's noise enters twice a night rather than every hour — and
    the nights are averaged, weighted by how long each was quiet.

    This is the idea behind continuous in-tank leak detection. It sidesteps
    what defeats a whole-month regression: within a delivery cycle, a leak, a
    meter error and a level-dependent chart error all grow together, and the
    month's data cannot pull them apart. At night the meter barely runs and
    the level barely moves.
    """
    measured = _net(month)
    hour = month["hour"]
    quiet = ((hour >= 23) | (hour <= 5)) & (month["delivery"] == 0)
    # At a 24-hour station the night is not empty, and every gallon sold
    # overnight carries the meter's error with it. The day says what that
    # error is; the night is corrected for it. The variance's slope on sales
    # is the meter error expressed as variance per gallon sold, which is
    # exactly the extra loss per gallon the night's balance would show.
    meter = meter_error(month) if correct_meter else 0.0
    rates, weights = [], []
    t = 0
    while t < HOURS:
        if not quiet[t]:
            t += 1
            continue
        start = t
        while t < HOURS and quiet[t]:
            t += 1
        end = t - 1
        span = end - start
        if span >= 4:
            sold = month["sales"][start + 1:end + 1].sum()
            lost = measured[start] - measured[end] - sold * (1 + meter)
            rates.append(lost / span)
            weights.append(span)
    if not rates:
        return float("nan")
    return float(np.average(rates, weights=weights))


def hourly_balance(month, level_terms: int = 0) -> float:
    """The balance hour by hour: each hour's unexplained loss is the leak
    plus a meter error proportional to that hour's sales,

        Δvariance(t) = L + m · sales(t) + noise,

    so a regression of hourly losses on hourly sales has the meter error as
    its slope and the leak as its intercept. It needs no quiet hours, only
    that sales vary from hour to hour — which they do at any station.

    A chart error enters the same way: as the level falls by what was sold,
    a chart wrong by e(level) adds e'(level) per gallon sold. It looks like a
    meter error that depends on the level. With level_terms > 0 the slope is
    allowed to vary with level (a polynomial of that degree), which is the
    one place a learned function belongs: correcting the gauge, never the
    leak, which stays a single explicit constant.

    Delivery hours, and the hour after, are left out.
    """
    variance, measured = _variance(month)
    change = np.diff(variance)
    sold = month["sales"][1:]
    keep = (month["delivery"][1:] == 0) & (month["delivery"][:-1] == 0)
    level = (measured[:-1] / CAPACITY - 0.5)[keep]
    columns = [np.ones(keep.sum()), sold[keep]]
    for k in range(1, level_terms + 1):
        columns.append(sold[keep] * level ** k)
    X = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(X, change[keep], rcond=None)
    return float(coef[0])


def conserving(month, steps: int = 300, seed: int = 0) -> float:
    """The same balance, with the chart error learned as a function of level.

    variance(t) = a + L·t + m·S(t) + d·D(t) − e(level(t))

    e is a small network of the level alone, centred so it cannot carry a
    constant. A leak grows with time while the level rises and falls with
    every delivery, so e cannot imitate L — that is the identifiability the
    structure provides, and it is measured below rather than assumed.
    """
    import torch

    torch.manual_seed(seed)
    variance, measured = _variance(month)
    t = torch.tensor(np.arange(HOURS) / HOURS, dtype=torch.float64)
    S = torch.tensor(np.cumsum(month["sales"]) / THROUGHPUT, dtype=torch.float64)
    D = torch.tensor(np.cumsum(month["delivery"]) / THROUGHPUT, dtype=torch.float64)
    level = torch.tensor(measured / CAPACITY, dtype=torch.float64).reshape(-1, 1)
    v = torch.tensor(variance, dtype=torch.float64)

    # Variable projection: for any chart curve the best leak, meter and
    # receipt terms are a least-squares solve, so they are solved exactly at
    # every step and only the chart network is learned by gradient. Learning
    # all of it by gradient started the leak at zero and stopped short of it:
    # measured, it estimated 0.05-0.08 gal/h for a 0.2 leak.
    X = torch.stack([torch.ones_like(t), t, S, D], dim=1)
    chart = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.Tanh(),
                                torch.nn.Linear(16, 16), torch.nn.Tanh(),
                                torch.nn.Linear(16, 1)).double()
    opt = torch.optim.Adam(chart.parameters(), lr=0.01)
    scale = float(np.std(variance)) or 1.0

    def solve():
        e = chart(level).reshape(-1) * scale
        e = e - e.mean()
        coef = torch.linalg.lstsq(X, (v + e).unsqueeze(1)).solution.reshape(-1)
        return coef, X @ coef - e

    for _ in range(steps):
        coef, pred = solve()
        loss = torch.mean((pred - v) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        coef, _ = solve()
    return float(coef[1]) / HOURS


def plain_network(month, reference, steps: int = 300, seed: int = 0) -> float:
    """Learn the hourly change from a previous leak-free month, then read the
    test month's mean residual as the leak. It has the same inputs as the
    others and no conservation law: nothing makes its residual mean a loss."""
    import torch

    torch.manual_seed(seed)

    def features(month):
        """Delivery hours are left out, as every SIR method leaves them out:
        the level jumps by thousands of gallons and a network's misfit there
        swamped everything — measured at +91 gal/h for a 0.2 leak."""
        measured = _net(month)
        change = np.diff(measured)
        X = np.column_stack([
            measured[:-1] / CAPACITY, month["sales"][1:] / 30,
            month["delivery"][1:] / 6000, (month["temp"][1:] - 60) / 10,
            np.diff(month["temp"]), np.sin(2 * math.pi * month["hour"][1:] / 24),
            np.cos(2 * math.pi * month["hour"][1:] / 24)])
        quiet = (month["delivery"][1:] == 0) & (month["delivery"][:-1] == 0)
        return (torch.tensor(X[quiet], dtype=torch.float64),
                torch.tensor(change[quiet], dtype=torch.float64))

    X, y = features(reference)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.Tanh(),
                              torch.nn.Linear(32, 32), torch.nn.Tanh(),
                              torch.nn.Linear(32, 1)).double()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    scale = float(y.std()) or 1.0
    for _ in range(steps):
        loss = torch.mean((net(X).reshape(-1) * scale - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    Xt, yt = features(month)
    with torch.no_grad():
        residual = yt - net(Xt).reshape(-1) * scale
    return float(-residual.mean())


# --------------------------------------------------------------------------
# the regulator's question
# --------------------------------------------------------------------------

def evaluate(months: int = 60, leaks=(0.0, 0.1, 0.2), seed: int = 0,
             methods=("classical", "quiet", "plain", "conserving"),
             progress=None, night: float = 0.08) -> Dict[str, Any]:
    """Each method, on the same simulated months, judged as 40 CFR 280.43(h)
    judges: threshold 0.1 gal/h (half of 0.2), false alarms at no leak,
    detection at 0.2 gal/h, and how far off the reported rate is.

    Also the minimum detectable leak at PD 0.95 / PFA 0.05, estimated as
    3.29 standard deviations of the no-leak estimate plus its bias — the
    standard reading for an unbiased, roughly normal estimator, and stated
    as an estimate because it assumes both.
    """
    rng = np.random.default_rng(seed)
    estimates: Dict[str, Dict[float, List[float]]] = {
        m: {L: [] for L in leaks} for m in methods}
    for L in leaks:
        for k in range(months):
            month_rng = np.random.default_rng(rng.integers(1 << 31))
            month = simulate(L, month_rng, night=night)
            if "classical" in methods:
                estimates["classical"][L].append(classical_sir(month))
            if "quiet" in methods:
                estimates["quiet"][L].append(quiet_hours(month))
            if "combined" in methods:
                estimates["combined"][L].append(quiet_hours(month, True))
            if "hourly" in methods:
                estimates["hourly"][L].append(hourly_balance(month))
            if "hourly_level" in methods:
                estimates["hourly_level"][L].append(hourly_balance(month, 2))
            if "conserving" in methods:
                estimates["conserving"][L].append(conserving(month, seed=k))
            if "plain" in methods:
                # the previous month at the same station: same tank, same
                # meters, same chart — and tight
                ref_rng = np.random.default_rng(month_rng.integers(1 << 31))
                reference = simulate(0.0, ref_rng, fixed=month["tank"],
                                     night=night)
                estimates["plain"][L].append(plain_network(month, reference, seed=k))
            if progress:
                progress({"leak": L, "month": k + 1, "of": months})

    threshold = 0.1
    report: Dict[str, Any] = {"months": months, "threshold": threshold,
                              "leaks": list(leaks), "methods": {}}
    for m in methods:
        e = {L: np.array(v) for L, v in estimates[m].items()}
        zero = e[leaks[0]]
        row = {
            "false_alarms": float(np.mean(np.abs(zero) >= threshold)),
            "detection": {str(L): float(np.mean(e[L] >= threshold))
                          for L in leaks if L > 0},
            "bias_at": {str(L): float(np.mean(e[L]) - L) for L in leaks},
            "spread_at": {str(L): float(np.std(e[L])) for L in leaks},
            "min_detectable": float(abs(np.mean(zero)) + 3.29 * np.std(zero)),
        }
        pd02 = row["detection"].get("0.2", 0.0)
        row["meets_standard"] = bool(row["false_alarms"] <= 0.05 and pd02 >= 0.95)
        report["methods"][m] = row
    return report
