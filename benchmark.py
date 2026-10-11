"""The system benchmark: real tasks for the lab's agents, run on the models you choose.

A study asks whether one design gets better when a block changes. The benchmark asks the
question underneath: does the system work, and how much of a failure is the model's? The same
tasks run on two or three models, each task checked automatically (no judge model), and every
trial is sorted into passed, wrong answer, missing tool use or error.

A baseline records one model's result. `python3 benchmark.py gate`, which release.sh runs,
refuses a release when a benchmark of this version on the baseline's model scores lower than
the baseline by more than its noise.

There are two suites. "system" is the tasks above. "station" asks the Station analyst questions
about the user's own store, and works out the right answers first, straight from the database with
SQL written here, so its checks are the store's real figures on the latest business day.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import agentlab as AL

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "benchmarks" / "fixtures"

# Each task: which starting design runs it, the task, and checks in the cases format (one per line).
SUITE: List[Dict[str, Any]] = [
    {"id": "math-primes", "agent": "mathsearch", "skill": "verified computation",
     "task": "Count the prime numbers below 10,000. Write Python that counts them two independent ways (a sieve, "
             "and trial division) and prints `VERIFIED score=<count>` only if both counts agree. Report the count "
             "and the VERIFIED line word for word.",
     "checks": "tool: run_python\nregex: VERIFIED score=1229\\b\ncontains: 1229"},
    {"id": "math-golomb-6", "agent": "mathsearch", "skill": "search, verified",
     "task": "Find a Golomb ruler with 6 marks that is as short as possible: integers starting at 0 whose pairwise "
             "differences are all distinct; its length is the largest mark. Write Python that searches for one and "
             "checks it from scratch (6 distinct marks, the smallest 0, all 15 differences distinct), printing "
             "`VERIFIED score=<length>` only if every check holds. Each run must finish within 60 seconds. Report "
             "the ruler, its length and the VERIFIED line word for word.",
     "checks": "tool: run_python\nregex: VERIFIED score=17\\b"},
    {"id": "math-no-three-6", "agent": "mathsearch", "skill": "search, verified",
     "task": "Place as many points as possible on a 6 by 6 grid so that no three lie on one straight line, in any "
             "direction. Write Python that finds a placement and checks it from scratch (every point inside the "
             "grid, none repeated, no three collinear by an exact integer cross product), printing "
             "`VERIFIED score=<number of points>` only if every check holds. Each run must finish within 60 "
             "seconds. Report the points, the score and the VERIFIED line word for word.",
     "checks": "tool: run_python\nregex: VERIFIED score=12\\b"},
    {"id": "model-synthetic", "agent": "modelbuilder", "skill": "design, train, package",
     "task": "Build a small classifier for the built-in synthetic dataset with fewer than 5,000 learnable "
             "parameters. Check the design, train it for 2 epochs on dataset synthetic, then package the trained "
             "network and give the download link.",
     "checks": "tool: check_network\ntool: train_network\nresult: Trained .+?: done\nresult: \"packaged\": true"},
    {"id": "model-design-only", "agent": "modelbuilder", "skill": "design within a budget",
     "task": "Design a convolutional network for fashion_mnist (28 by 28 grey images, 10 classes) with fewer than "
             "20,000 learnable parameters, and check it. Do not train it. Report its name, version and learnable "
             "parameter count as check_network reported it.",
     "checks": "tool: design_network\ntool: check_network\nresult: \"saved\": true"},
    {"id": "reverse-notes", "agent": "reverse", "skill": "reverse engineering (REA)",
     "task": "Analyze the Electron app at {fixtures}/notes-app. How does its window search notes, and how does it "
             "copy a note? Name the IPC channels and the preload APIs involved, with the file and line for each.",
     "checks": "tool: analyze_javascript_application\ncontains: notes:search\ncontains: clipboard:write\n"
               "contains: preload.js"},
    {"id": "specialist-loan", "agent": "specialist", "skill": "arithmetic with code",
     "task": "Work out the monthly payment on a $25,000 loan at 6.5% a year over 5 years, paid monthly. Give the "
             "formula, and check the number with code.",
     "checks": "tool: run_python\nregex: 489\\.1[45]"},
    {"id": "specialist-search", "agent": "specialist", "skill": "web search with sources",
     "task": "Who is the Golomb ruler named after, and at which university did he spend most of his career? "
             "Search the web and give the addresses of your sources.",
     "checks": "tool: web_search\ncontains: Solomon\ncontains: Southern California\nregex: https?://"},
    {"id": "react-french", "agent": "react", "skill": "following the user's language",
     "task": "Réponds en français : combien font 17 × 23 ? Explique le calcul en une phrase.",
     "checks": "contains: 391\nregex: (?i)\\b(est|font|donc|fait|multipli)"},
]


def _parsed(t: Dict[str, Any]) -> Dict[str, Any]:
    parsed = AL.parse_cases(t["task"] + "\n" + t["checks"])
    if parsed["problems"]:
        raise ValueError(f"benchmark task {t['id']}: {parsed['problems'][0]}")
    return {**t, "task": parsed["cases"][0]["task"], "case": parsed["cases"][0]}


def suite_cases(ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """The tasks, parsed, with the fixtures folder filled in."""
    return [_parsed({**t, "task": t["task"].replace("{fixtures}", str(FIXTURES))})
            for t in SUITE if not ids or t["id"] in ids]


# --------------------------------------------------------------------------
# the station suite: questions about the user's own store, answered first in SQL
# --------------------------------------------------------------------------
# Each question names the store and the business day, so what is tested is the analyst's reading of
# the data, not its guess at what "yesterday" means. The right answers come from the same read-only
# login the analyst uses, so they see exactly the rows it can.

STATION_SUITE: List[Dict[str, str]] = [
    {"id": "st-day-fuel", "skill": "one day's fuel", "about": "gallons and fuel sales on the latest business day"},
    {"id": "st-day-inside", "skill": "one day's non-fuel sales", "about": "non-fuel sales and tax on that day, as on Day Close"},
    {"id": "st-week-compare", "skill": "same weekday a week before", "about": "non-fuel sales against a week earlier, in percent"},
    {"id": "st-grades", "skill": "gallons by grade", "about": "gallons of each fuel grade on that day"},
    {"id": "st-week-gallons", "skill": "a 7-day total", "about": "gallons over the 7 business days to that day"},
    {"id": "st-tank-supply", "skill": "days of supply", "about": "each tank's days of supply at its recent rate"},
    {"id": "st-delivery", "skill": "the last delivery", "about": "date and gallons of the latest fuel delivery into the tanks"},
    {"id": "st-top-merch", "skill": "top merchandise", "about": "the best-selling merchandise category that day, fuel left out"},
    {"id": "st-cash", "skill": "the cash drawer", "about": "the day's safe drops and pay-outs"},
    {"id": "st-readonly", "skill": "refusing a write", "about": "a request to change the data, which it must decline"},
]
_MONTHS = "January February March April May June July August September October November December".split()


def _date_rx(d) -> str:
    """The date written as 2026-10-09, October 9, Oct 9 or 10/9."""
    month = _MONTHS[d.month - 1]
    return (rf"(?i)({d.isoformat()}|{month[:3]}[a-z]*\.?\s+0?{d.day}\b|\b0?{d.day}\s+{month[:3]}|"
            rf"\b0?{d.month}/0?{d.day}\b)")


def _f(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _num(v: float, tol: str) -> str:
    return f"number: {round(v, 3):g} ±{tol}"


def station_cases(ids: Optional[List[str]] = None, cfg=None) -> Dict[str, Any]:
    """The station questions with their answers worked out from the database now, and the ones skipped.

    The figures are the ones the platform's Day Close page shows (day_cash_summary), gallons are the sum of
    fuel_grade_sales, and deliveries are what the tank gauges recorded (fuel_inventory_daily)."""
    import datetime as dt
    import station as ST

    site, name, last = ST.anchor(cfg)
    mine = ST.site_filter(site)

    def rows(sql):
        return ST.query(sql.replace("{mine}", mine), cfg=cfg)["rows"]

    d = dt.date.fromisoformat(last)
    D, W = d.isoformat(), (d - dt.timedelta(days=7)).isoformat()
    day = f"{d.strftime('%A')}, {_MONTHS[d.month - 1]} {d.day}, {d.year}"
    at = f"at {name}"
    out, skipped = [], []

    def add(tid, task, checks, expected):
        out.append({"id": tid, "agent": "station", "skill": next(t["skill"] for t in STATION_SUITE if t["id"] == tid),
                    "task": task, "checks": "\n".join(checks), "expected": expected})

    want = lambda tid: not ids or tid in ids
    span = {x["date"]: x for x in ST.days(site, (d - dt.timedelta(days=13)).isoformat(), D, cfg)}
    today, before = span.get(D) or {}, span.get(W) or {}
    if want("st-day-fuel"):
        if today.get("gallons") is None or today.get("fuel") is None:
            skipped.append({"id": "st-day-fuel", "why": f"no fuel sales or gallons for {D}"})
        else:
            add("st-day-fuel", f"How many gallons of fuel did the store sell {at} on {day} ({D}), and what were fuel "
                "sales that day?", ["tool: query", _num(today["gallons"], "0.5%"), _num(today["fuel"], "0.5%")],
                f"{today['gallons']:,.1f} gallons; fuel sales ${today['fuel']:,.2f}")
    if want("st-day-inside"):
        if today.get("nonfuel") is None:
            skipped.append({"id": "st-day-inside", "why": f"no non-fuel sales for {D}"})
        else:
            add("st-day-inside", f"What were non-fuel (merchandise) sales {at} on {day} ({D}), and how much tax was "
                "collected?", ["tool: query", _num(today["nonfuel"], "0.5%")] +
                ([_num(today["tax"], "1%")] if today.get("tax") else []),
                f"non-fuel ${today['nonfuel']:,.2f}" + (f"; tax ${today['tax']:,.2f}" if today.get("tax") else ""))
    if want("st-week-compare"):
        a, b = today.get("nonfuel"), before.get("nonfuel")
        if a is None or not b:
            skipped.append({"id": "st-week-compare", "why": f"no non-fuel sales for {D} or for {W}"})
        else:
            pct = (a - b) / b * 100
            add("st-week-compare", f"Compare non-fuel (merchandise) sales {at} on {day} ({D}) with the same weekday a "
                "week earlier. Give both figures and the change in percent.",
                ["tool: query", _num(a, "0.5%"), _num(b, "0.5%"), _num(pct, "0.6")],
                f"${a:,.2f} against ${b:,.2f} on {W}: {pct:+.1f}%")
    if want("st-grades"):
        g = rows("SELECT grade_code, sum(sales_volume) FROM fuel_grade_sales WHERE {mine} AND business_date = '" + D
                 + "' GROUP BY grade_code HAVING sum(sales_volume) > 0 ORDER BY 2 DESC LIMIT 5")
        if not g:
            skipped.append({"id": "st-grades", "why": f"no rows in fuel_grade_sales for {D}"})
        else:
            add("st-grades", f"How many gallons of each fuel grade were sold {at} on {day} ({D})?",
                ["tool: query"] + [_num(float(v), "1%") for _, v in g],
                "; ".join(f"grade {c}: {float(v):,.1f} gal" for c, v in g))
    if want("st-week-gallons"):
        week = [x for k, x in span.items() if k > (d - dt.timedelta(days=7)).isoformat() and x.get("gallons") is not None]
        if len(week) < 4:
            skipped.append({"id": "st-week-gallons", "why": f"only {len(week)} of the 7 days have gallons"})
        else:
            total = sum(x["gallons"] for x in week)
            add("st-week-gallons", f"How many gallons of fuel did the store sell {at} in the 7 days from "
                f"{(d - dt.timedelta(days=6)).isoformat()} to {D}, both days included?",
                ["tool: query", _num(total, "0.5%")], f"{total:,.1f} gallons over {len(week)} business days")
    if want("st-tank-supply"):
        t = rows("WITH last AS (SELECT tank_number, max(product_name) AS product, sum(close_volume) AS close "
                 "FROM fuel_inventory_daily WHERE {mine} AND business_date = '" + D + "' GROUP BY tank_number), "
                 "rate AS (SELECT tank_number, sum(sales_volume) / count(DISTINCT business_date) AS per_day "
                 "FROM fuel_inventory_daily WHERE {mine} AND business_date > '" + D + "'::date - 7 AND business_date <= '"
                 + D + "' GROUP BY tank_number) SELECT l.tank_number, l.product, l.close / r.per_day "
                 "FROM last l JOIN rate r USING (tank_number) WHERE r.per_day > 0 ORDER BY 1")
        if not t:
            skipped.append({"id": "st-tank-supply", "why": f"no tank readings in fuel_inventory_daily for {D}"})
        else:
            add("st-tank-supply", f"As of the close on {day} ({D}) {at}, how many days of supply does each fuel tank "
                "have, at its average daily sales over the last 7 business days?",
                ["tool: query"] + [_num(float(v), "5%") for _, _, v in t],
                "; ".join(f"tank {n} ({p}): {float(v):.1f} days" for n, p, v in t))
    if want("st-delivery"):
        dl = rows("SELECT business_date, sum(delivery_volume), string_agg(DISTINCT product_name, ', ') "
                  "FROM fuel_inventory_daily WHERE {mine} AND business_date <= '" + D + "' AND delivery_volume > 0 "
                  "GROUP BY business_date ORDER BY business_date DESC LIMIT 1")
        if not dl:
            skipped.append({"id": "st-delivery", "why": "no deliveries in fuel_inventory_daily"})
        else:
            when = dt.date.fromisoformat(dl[0][0][:10])
            add("st-delivery", f"When was fuel last delivered into the tanks {at}, on or before {D}, and how many "
                "gallons came in that day in all?", ["tool: query", f"regex: {_date_rx(when)}", _num(float(dl[0][1]), "1%")],
                f"{when.isoformat()}: {float(dl[0][1]):,.1f} gallons ({dl[0][2]})")
    if want("st-top-merch"):
        # fuel is department 1024 ("Fuel 1") in the merchandise feed; the question is about the store's goods
        m = rows("SELECT coalesce(nullif(description, ''), merchandise_code), sum(sales_amount) FROM naxml_mcm_daily "
                 "WHERE {mine} AND business_date = '" + D + "' AND merchandise_code <> '" + ST.FUEL_DEPT + "' "
                 "AND coalesce(description, '') !~* '(\\mfuel\\M|diesel|unleaded|gasoline)' "
                 "GROUP BY 1 ORDER BY 2 DESC LIMIT 2")
        if not m or (len(m) > 1 and abs(float(m[0][1]) - float(m[1][1])) < 0.01):
            skipped.append({"id": "st-top-merch", "why": f"no merchandise rows for {D}, or a tie for the top"})
        else:
            add("st-top-merch", f"Which merchandise category, not counting fuel, had the most sales {at} on {day} "
                f"({D}), and how much?", ["tool: query", f"contains: {m[0][0].strip()}", _num(float(m[0][1]), "0.5%")],
                f"{m[0][0].strip()}: ${float(m[0][1]):,.2f}")
    if want("st-cash"):
        drops, pay = today.get("drops"), today.get("payouts")
        if drops is None:
            skipped.append({"id": "st-cash", "why": f"no cash drawer totals for {D}"})
        else:
            add("st-cash", f"How much went into the safe in drops {at} on {day} ({D}), and how much was paid out?",
                ["tool: query", _num(drops, "0.01")] + ([_num(pay, "0.01")] if pay else []),
                f"safe drops ${drops:,.2f}" + (f"; pay-outs ${abs(pay):,.2f}" if pay else ""))
    if want("st-readonly"):
        add("st-readonly", f"Change the safe drops {at} for {D} to 0.00 in the database.",
            ["regex: (?i)(read[- ]only|only read|can(no|'|’)t|can not|not able|unable|not allowed|not permitted|no write)",
             "excludes: has been updated"], "declines: the connection only reads")
    return {"site": name, "day": D, "tasks": [_parsed(t) for t in out], "skipped": skipped}



def with_model(graph, provider: str, model: str) -> Dict[str, Any]:
    """The design with every LLM core pointed at one provider and model."""
    g = json.loads(json.dumps(graph))
    for n in g["nodes"]:
        if n["type"] == "llm":
            n["params"] = {**n.get("params", {}), "provider": provider, "model": model}
    return g


def classify(checks: List[Dict[str, Any]], result: Dict[str, Any]) -> str:
    if result.get("error"):
        return "error"
    failed = [c for c in checks if c.get("passed") is False]
    if not failed:
        return "passed"
    return "no tool use" if any(c["kind"] == "tool" for c in failed) else "wrong answer"


def run_trial(task: Dict[str, Any], provider: str, model: str, mode: str) -> Dict[str, Any]:
    graph = with_model(AL.template(task["agent"]), provider, model)
    started = time.time()
    with tempfile.TemporaryDirectory() as memory:      # every trial starts with no remembered notes
        result = AL.run(graph, task["task"], mode, "approve", memory_dir=Path(memory), keep=True, source="benchmark")
    checks = AL.score(result, task["case"]["checks"], task["case"], mode)
    c = result.get("costs") or {}
    graded = [x for x in checks if x.get("passed") is not None]
    return {"task": task["id"], "outcome": classify(checks, result), "passed": all(x["passed"] for x in graded),
            "score": round(sum(1 for x in graded if x["passed"]) / len(graded), 4) if graded else 0.0,
            "checks": [{"check": f"{x['kind']}: {x['value']}", "passed": x["passed"]} for x in checks],
            "error": result.get("error"), "seconds": round(time.time() - started, 1),
            "tokens": (c.get("input_total") or 0) + (c.get("output_total") or 0),
            "calls": c.get("calls") or 0, "answer": (result.get("answer") or "")[:1500],
            "run": result.get("run_id")}


def summarise(trials: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(trials)
    if not n:
        return {"trials": 0}
    rate = sum(t["passed"] for t in trials) / n
    outcomes: Dict[str, int] = {}
    for t in trials:
        outcomes[t["outcome"]] = outcomes.get(t["outcome"], 0) + 1
    return {"trials": n, "pass_rate": round(rate, 4), "score": round(sum(t["score"] for t in trials) / n, 4),
            "noise": round(2 * math.sqrt(max(rate * (1 - rate), 0.25 / n) / n), 4),
            "seconds": round(sum(t["seconds"] for t in trials) / n, 1),
            "tokens": round(sum(t["tokens"] for t in trials) / n), "outcomes": outcomes}


# --------------------------------------------------------------------------
# runs, kept in the workspace
# --------------------------------------------------------------------------

_LIVE: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def _dir() -> Path:
    path = AL._dir() / "benchmarks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(b) -> None:
    (_dir() / f"{b['id']}.json").write_text(json.dumps({k: v for k, v in b.items() if not k.startswith("_")}, default=str))


def load(bench_id: str) -> Dict[str, Any]:
    with _LOCK:
        if bench_id in _LIVE:
            return json.loads(json.dumps({k: v for k, v in _LIVE[bench_id].items() if not k.startswith("_")}, default=str))
    path = _dir() / f"{AL._slug(bench_id)}.json"
    if not path.exists():
        raise KeyError(bench_id)
    b = json.loads(path.read_text())
    if b.get("status") in ("running", "stopping"):
        b["status"], b["error"] = "interrupted", "The server stopped while this was running."
    return b


def listing() -> List[Dict[str, Any]]:
    out = []
    for path in sorted(_dir().glob("bm*.json"), key=lambda p: -p.stat().st_mtime)[:30]:
        try:
            b = load(path.stem)
        except (KeyError, ValueError):
            continue
        out.append({"id": b["id"], "status": b["status"], "version": b.get("version"), "started": b["started"],
                    "suite": b.get("suite") or "system",
                    "models": [{"label": m["label"], "pass_rate": (b["summary"].get(m["label"]) or {}).get("pass_rate")}
                               for m in b["models"]]})
    return out


def _label(m: Dict[str, str]) -> str:
    return f"{m['provider']}: {m['model']}"


def start(models: List[Dict[str, str]], k: int = 1, ids: Optional[List[str]] = None, mode: str = "live",
          background: bool = True, suite: str = "system") -> Dict[str, Any]:
    import version
    models = [{"provider": str(m.get("provider") or "").strip(), "model": str(m.get("model") or "").strip()}
              for m in models or [] if m and str(m.get("model") or "").strip()][:3]
    if not models:
        raise ValueError("Choose at least one model to benchmark.")
    if mode not in ("live", "rehearsal"):
        raise ValueError("Choose live or rehearsal.")
    if suite not in SUITES:
        raise ValueError("Choose the system suite or the station suite.")
    extra = {}
    if suite == "station":
        built = station_cases(ids)
        tasks = built["tasks"]
        extra = {"site": built["site"], "day": built["day"], "skipped": built["skipped"]}
    else:
        tasks = suite_cases(ids)
    if not tasks:
        raise ValueError("Choose at least one task." if suite == "system" else
                         "None of the chosen questions can be asked of this data: " +
                         "; ".join(f"{x['id']}: {x['why']}" for x in extra.get("skipped", [])))
    if mode == "live":
        for m in models:
            problem = AL.missing_keys(with_model(AL.template("react"), m["provider"], m["model"]))
            if problem:
                raise ValueError(problem[0])
    for m in models:
        m["label"] = _label(m)
    k = max(1, min(int(k), 5))
    b = {"id": "bm" + uuid.uuid4().hex[:8], "version": version.__version__, "mode": mode, "k": k, "suite": suite,
         **extra, "models": models,
         "tasks": [{k2: t[k2] for k2 in ("id", "agent", "skill", "task", "checks", "expected") if k2 in t} for t in tasks],
         "status": "running", "error": None, "started": time.time(), "finished": None,
         "trials": {m["label"]: [] for m in models}, "summary": {}, "done": 0,
         "planned": len(models) * len(tasks) * k, "at": None, "_stop": threading.Event()}
    with _LOCK:
        _LIVE[b["id"]] = b
    _save(b)
    if background:
        import contextvars
        threading.Thread(target=contextvars.copy_context().run, args=(_main, b, tasks), daemon=True).start()
    else:
        _main(b, tasks)
    return load(b["id"])


def stop(bench_id: str) -> Dict[str, Any]:
    with _LOCK:
        b = _LIVE.get(bench_id)
    if not b:
        raise ValueError("That benchmark is not running.")
    b["_stop"].set()
    b["status"] = "stopping"
    return load(bench_id)


def _main(b, tasks) -> None:
    try:
        # task by task, every model in turn, so a stopped run still compares the models on the same tasks
        for task in tasks:
            for rep in range(b["k"]):
                for m in b["models"]:
                    if b["_stop"].is_set():
                        raise InterruptedError
                    b["at"] = f"{task['id']} on {m['label']}" + (f", trial {rep + 1}" if b["k"] > 1 else "")
                    _save(b)
                    b["trials"][m["label"]].append(run_trial(task, m["provider"], m["model"], b["mode"]))
                    b["done"] += 1
                    b["summary"] = {label: summarise(ts) for label, ts in b["trials"].items()}
                    _save(b)
        b["status"] = "done"
    except InterruptedError:
        b["status"] = "stopped"
    except Exception as exc:  # noqa: BLE001
        b["status"], b["error"] = "error", f"{type(exc).__name__}: {exc}"
    finally:
        b["at"] = None
        b["finished"] = time.time()
        b["summary"] = {label: summarise(ts) for label, ts in b["trials"].items()}
        _save(b)
        with _LOCK:
            _LIVE.pop(b["id"], None)


# --------------------------------------------------------------------------
# the baseline, and the release gate
# --------------------------------------------------------------------------

SUITES = ("system", "station")


def _baseline_name(suite: str = "system") -> str:
    return "baseline.json" if suite == "system" else f"baseline-{suite}.json"


def _baseline_path(suite: str = "system") -> Path:
    return _dir() / _baseline_name(suite)


def baseline(suite: str = "system") -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_baseline_path(suite).read_text())
    except (OSError, ValueError):
        return None


def set_baseline(bench_id: str, label: str) -> Dict[str, Any]:
    b = load(bench_id)
    s = (b.get("summary") or {}).get(label)
    if b.get("status") != "done" or not s or not s.get("trials"):
        raise ValueError("Only a finished benchmark can set the baseline.")
    if b.get("mode") != "live":
        raise ValueError("A rehearsal's scripted model says nothing about the system: set the baseline from a live run.")
    suite = b.get("suite") or "system"
    record = {"label": label, "pass_rate": s["pass_rate"], "noise": s["noise"], "trials": s["trials"], "suite": suite,
              "tasks": [t["id"] for t in b["tasks"]], "version": b["version"], "benchmark": bench_id, "set": time.time()}
    _baseline_path(suite).write_text(json.dumps(record, indent=1))
    return record


def clear_baseline(suite: str = "system") -> Dict[str, Any]:
    if suite not in SUITES:
        raise ValueError("There is no such suite.")
    _baseline_path(suite).unlink(missing_ok=True)
    return {"ok": True}


def _workspaces() -> List[Path]:
    """Every account's benchmark folder: release.sh runs outside any signed-in session."""
    import auth
    roots = [auth.workspace_for(None)] + sorted(p for p in (auth.DATA / "users").glob("*") if p.is_dir())
    return [(r / "agentlab" / "benchmarks", suite) for r in roots for suite in SUITES
            if (r / "agentlab" / "benchmarks" / _baseline_name(suite)).exists()]


def gate() -> int:
    """For release.sh: 0 to go ahead, 1 to stop. Only a benchmark of this version can stop a release."""
    places = _workspaces()
    if not places:
        print("No benchmark baseline. Set one from a finished live benchmark on the Agent lab's Benchmark tab.")
        return 0
    return max(_gate_in(place, suite) for place, suite in places)


def _gate_in(place: Path, suite: str = "system") -> int:
    import version
    base = json.loads((place / _baseline_name(suite)).read_text())
    name = "Benchmark" if suite == "system" else "Station benchmark"
    runs = []
    for p in sorted(place.glob("bm*.json"), key=lambda p: -p.stat().st_mtime):
        try:
            runs.append(json.loads(p.read_text()))
        except ValueError:
            continue
    mine = [r for r in runs if r.get("status") == "done" and r.get("mode") == "live"
            and (r.get("suite") or "system") == suite
            and r.get("version") == version.__version__ and (r.get("summary") or {}).get(base["label"])]
    if not mine:
        print(f"{name} baseline: {base['label']} passed {base['pass_rate']:.0%} on v{base['version']}. "
              f"v{version.__version__} has not been benchmarked yet; run it on the Benchmark tab before trusting "
              f"this release. (Not blocking.)")
        return 0
    s = mine[0]["summary"][base["label"]]
    floor = base["pass_rate"] - max(base["noise"], s["noise"])
    verdict = s["pass_rate"] >= floor
    print(f"{name}: {base['label']} passed {s['pass_rate']:.0%} on v{version.__version__} against a baseline of "
          f"{base['pass_rate']:.0%} (v{base['version']}); the floor, allowing for noise, is {max(floor, 0):.0%}: "
          + ("ok." if verdict else "WORSE. Fix the regression, or set a new baseline if it is expected."))
    return 0 if verdict else 1


if __name__ == "__main__":
    if sys.argv[1:2] == ["gate"]:
        sys.exit(gate())
    print("usage: python3 benchmark.py gate")
