"""The super agent: rules that hold by construction, a comparison that cannot flatter, and a run from a goal
to a model on the canvas.

Run with `python tests/test_superagent.py`. Everything here needs torch; without it the checks skip.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["AGENTLAB_NO_PUSH"] = "1"

import blockloader  # noqa: E402
import codegen  # noqa: E402
import graph as G  # noqa: E402

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

PASSED, FAILED = [], []
KEEP = {}


def check(name):
    def wrap(fn):
        if os.environ.get("ONLY") and os.environ["ONLY"] not in name:
            return fn
        try:
            fn()
            PASSED.append(name)
            print(f"  pass  {name}")
        except AssertionError as exc:
            FAILED.append((name, str(exc)))
            print(f"  FAIL  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            FAILED.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        return fn
    return wrap


if HAVE_TORCH:
    import rulesource
    import superagent as SA

    blockloader.load_all()
    torch.set_num_threads(2)
    SA._load_words()
    SA.ARXIV_PAUSE = 0.0

    @check("every rule holds exactly with random weights, in every combination the super agent would build")
    def _():
        built = 0
        for tid, make in SA.TASKS.items():
            task = make()
            for rules in SA.candidates_for(task, {r: 1 for r in task["holds"]}, 8):
                try:
                    model = SA.make_model(task, rules, 16)
                except ValueError:
                    continue
                built += 1
                for r in rules:
                    v = SA.violation(r, model, task)
                    assert v is None or v < SA.EXACT, f"{tid}: {'+'.join(rules)} breaks {r} by {v:.3g}"
            # and the conventional network does break them, so the check can tell the difference
            plain = SA.make_plain(task)
            broke = [SA.violation(r, plain, task) for r in SA.allowed_rules(task)]
            assert any(v is not None and v > 0.01 for v in broke), f"{tid}: the plain network broke nothing: {broke}"
        assert built >= 40, built

    @check("a motion is always read off an energy, and symmetric energies conserve what Noether says they do")
    def _():
        task = SA.TASKS["two_body"]()
        for rules in SA.candidates_for(task, {}, 8):
            assert rules[0] == "energy", rules
        # translation-invariant energy: the forces sum to zero, so total momentum cannot change
        model = SA.make_model(task, ["energy", "permutation", "translation"], 16)
        z = task["sample"](64, torch.Generator().manual_seed(1))
        dp = SA._run(model, z)[..., 2:].sum(dim=1)
        assert float(dp.abs().max()) < 1e-4, float(dp.abs().max())
        # rotation-invariant energy: the torques sum to zero, so angular momentum cannot change
        model = SA.make_model(task, ["energy", "rotation"], 16)
        f = SA._run(model, z)
        q, p, dq, dpp = z[..., :2], z[..., 2:], f[..., :2], f[..., 2:]
        dL = (dq[..., 0] * p[..., 1] - dq[..., 1] * p[..., 0] + q[..., 0] * dpp[..., 1] - q[..., 1] * dpp[..., 0]).sum(dim=1)
        assert float(dL.abs().max()) < 1e-4, float(dL.abs().max())

    @check("combinations that cannot be true are refused by name, and a rule with nothing to compare is not offered")
    def _():
        Net = rulesource.build("RuleNet")
        for kw, word in [(dict(shape=[3], rules=["monotone", "reflection"]), "even"),
                         (dict(shape=[3], rules=["energy"]), "rows"),
                         (dict(shape=[2, 2], rules=["convex"]), "one row"),
                         (dict(shape=[2, 1], rules=["rotation"], k=1), "k >= 2"),
                         (dict(shape=[2, 3], rules=["energy"], k=1), "numbers"),
                         (dict(shape=[3], rules=["telepathy"]), "no rule")]:
            try:
                Net(**kw)
                raise AssertionError(f"{kw} was built")
            except ValueError as exc:
                assert word in str(exc), (kw, str(exc))
        assert "reflection" not in SA.allowed_rules(SA.TASKS["demand"]())       # prices are never negative: no mirror image
        assert "reflection" in SA.allowed_rules(SA.TASKS["cost"]())
        assert "monotone" not in SA.allowed_rules(SA.TASKS["cost"]())           # nobody said which way each input pushes
        assert SA.allowed_rules(SA.TASKS["pendulum"]()) == ["energy", "reflection"]

    @check("a candidate wins a size only beyond both spreads, and the conventional network is never the smaller one")
    def _():
        base = [{"n": 16, "error": 1.0, "spread": 0.1}, {"n": 64, "error": 0.5, "spread": 0.05}]
        ahead = [{"n": 16, "error": 0.5, "spread": 0.1}, {"n": 64, "error": 0.2, "spread": 0.05}]
        close = [{"n": 16, "error": 0.85, "spread": 0.1}, {"n": 64, "error": 0.62, "spread": 0.05}]
        a, c = SA.compare(ahead, base), SA.compare(close, base)
        assert a["wins"] == [16, 64] and not a["losses"] and a["gain"] > 2 and a["reaches_at"] == 16 and a["data_saving"] == 4.0
        assert c["wins"] == [] and c["losses"] == [64] and c["reaches_at"] is None, c
        task = SA.TASKS["pair_energy"]()
        big = SA.count(SA.make_model(task, ["permutation"], 64))
        assert SA.count(SA.make_plain(task, at_least=big)) >= big
        assert SA.count(SA.make_plain(task, at_least=big, act="tanh")) >= big

    @check("the brief reads the kind of problem from the goal, and Radar can supply the goal")
    def _():
        for goal, kind, first in [("a fast model of how a planet orbits a star", "dynamics", "orbit"),
                                  ("predict fuel demand from price", "tables", "demand"),
                                  ("a property of a point cloud whatever the order", "sets", None),
                                  ("which item was hidden in a long list", "sequences", "hidden_item")]:
            b = SA.brief(goal, 2, live=False)
            assert b["kind"] == kind and (first is None or b["tasks"][0] == first), (goal, b)
        import radar
        real = radar.view
        radar.view = lambda: {"items": [{"headline": "A new cooking app", "what": "recipes", "url": "u1"},
                                        {"headline": "Hamiltonian networks conserve energy in orbit prediction",
                                         "what": "a symplectic model of planetary motion", "url": "u2"}]}
        try:
            got = SA.from_radar()
            assert got["source"]["url"] == "u2" and "Hamiltonian" in got["goal"], got
            radar.view = lambda: {"items": []}
            try:
                SA.from_radar()
                raise AssertionError("an empty radar gave a goal")
            except ValueError as exc:
                assert "Radar" in str(exc)
        finally:
            radar.view = real

    class _Hit:
        def __init__(self, i, title, abstract):
            self.id, self.title, self.abstract, self.year, self.authors, self.source, self.pdf_url = f"2501.{i:05d}", title, abstract, "2025", "A. Author", "arxiv", ""

    PAPERS = [_Hit(1, "Symmetric energies for interacting bodies",
                   "We learn a Hamiltonian so that energy is conserved. The energy is rotation invariant, hence angular "
                   "momentum is conserved, and identical particles make it permutation invariant."),
              _Hit(2, "Friction matters", "Real systems dissipate. We model friction with a port-Hamiltonian damping matrix.")]

    @check("a run goes from a goal to a measured model on the canvas: swarm, cards, verify, train, slim, hand over")
    def _():
        home = Path(tempfile.mkdtemp())
        real = (SA.home, SA._search, SA._ask, SA._live)
        asked = []

        def ask(system, prompt, max_tokens=0):
            asked.append(system[:40])
            if "mock tasks" in system:
                return {"tasks": ["two_body", "nonsense"], "why": "Two bound bodies is the goal itself.", "queries": ["rotation invariant Hamiltonian"]}
            return {"cards": [{"paper": 1, "rule": "energy", "statement": "Energy is conserved.", "quote": "We learn a Hamiltonian so that energy is conserved."},
                              {"paper": 1, "rule": "rotation", "statement": "The energy does not change under rotation.", "quote": "The energy is rotation invariant"},
                              {"paper": 2, "rule": "other", "name": "Dissipation", "statement": "Energy only falls.", "quote": "Real systems dissipate.",
                               "build": "Add a positive semi-definite damping matrix."},
                              {"paper": 9, "rule": "energy", "quote": "no such paper"}, {"paper": 1, "rule": "energy", "quote": ""}]}
        SA.home, SA._search, SA._ask, SA._live = (lambda: home), (lambda q, n: PAPERS), ask, (lambda: True)
        try:
            run = SA.start("a small fast model of how two bound bodies move", "quick", background=False)
            d = SA.read(run["id"])
            assert d["status"] == "done" and d["mode"] == "live" and d["reachable"], (d["status"], d["error"])
            assert d["brief"]["tasks"] == ["two_body"] and d["brief"]["by"] == "model"
            assert len(d["swarm"]) >= 4 and all(a["status"] == "done" and a["papers"] for a in d["swarm"])
            rules = {r["id"]: r for r in d["rules"]}
            assert rules["energy"]["from"] == "reading" and rules["rotation"]["evidence"] >= 1 and rules["translation"]["from"] == "library"
            assert all(c["quote"] and c["url"].startswith("https://arxiv.org/abs/") for c in d["cards"])      # cards without a quote or a paper are dropped
            assert [c["name"] for c in d["unbuilt"]] == ["Dissipation"]                                       # read, and said not to be buildable yet
            T = d["tasks"][0]
            assert T["checks"] and all(c["ok"] for c in T["checks"])
            assert T["candidates"][0]["rules"] == ["energy", "rotation"], T["candidates"][0]["rules"]         # what was read about together comes first
            assert T["baseline"]["params"] >= max(c["params"] for c in T["candidates"])
            win = T["candidates"][T["winner"]]
            assert T["winner"] is not None and "energy" in win["rules"] and win["gain"] > 1.5 and not win["losses"], win
            assert T["slim"]["params"] <= win["params"] and T["slim"]["smaller"] > 1
            assert d["won"] == 1 and "won on 1 of 1" in d["verdict"] and "papers were looked at" in d["reading"]
            # the hand-over: a design that analyses and runs, a file that builds, and weights that fit it
            g = SA.to_canvas(d["id"])
            parsed = G.parse(g)
            rep = G.analyze(parsed)
            assert rep["ok"], rep["errors"]
            space = {}
            exec(codegen.to_pytorch(parsed, rep), space)
            z = SA.TASKS["two_body"]()["sample"](3, torch.Generator().manual_seed(0))
            assert tuple(space[g["name"]]()(z).shape) == (3, 2, 4)
            file_space = {"__name__": "generated"}
            exec(SA.model_file(d["id"]), file_space)
            model = file_space["build"]()
            model.load_state_dict(torch.load(home / f"{d['id']}-two_body.pt")["state"])
            assert SA.count(model) == T["slim"]["params"]
            assert SA.listing()[0]["id"] == d["id"] and SA.listing()[0]["won"] == 1
            json.dumps(d)
            KEEP["won"], KEEP["home"] = d, home
        finally:
            SA.home, SA._search, SA._ask, SA._live = real

    @check("with no model and no search it still runs, and says no paper was read; with papers and no model it matches by vocabulary")
    def _():
        home = Path(tempfile.mkdtemp())
        real = (SA.home, SA._search, SA._live)

        def down(q, n):
            raise OSError("no route")
        SA.home, SA._search, SA._live = (lambda: home), down, (lambda: False)
        try:
            d = SA.read(SA.start("which item is hidden in a long list", "quick", background=False)["id"])
            assert d["status"] == "done" and d["reachable"] is False and d["mode"] == "rehearsal"
            assert all(a["status"] == "unreachable" for a in d["swarm"]) and not d["cards"]
            assert all(r["from"] == "library" for r in d["rules"]) and "No paper was read" in d["reading"]
            assert d["tasks"][0]["id"] == "hidden_item" and d["tasks"][0]["baseline"]["kind"] == "GRU"
            got = SA.read_angle("goal", "symmetry", "q", SA.BUDGETS["quick"], live=False)
            assert got["reachable"] is False
            SA._search = lambda q, n: PAPERS
            got = SA.read_angle("goal", "symmetry", "q", SA.BUDGETS["quick"], live=False)
            found = {c["rule"] for c in got["cards"]}
            assert {"energy", "rotation", "permutation"} <= found and all(c["by"] == "vocabulary" for c in got["cards"]), found
            # one run at a time, and a goal is required unless it is asked to pick
            for kw, word in [(dict(goal=""), "Type a goal"), (dict(goal="x", budget="lavish"), "budget")]:
                try:
                    SA.start(**kw)
                    raise AssertionError(f"{kw} started")
                except ValueError as exc:
                    assert word in str(exc), str(exc)
        finally:
            SA.home, SA._search, SA._live = real

    def _lost_run():
        """A run in which the rule was true and still did not pay: ahead with little data, behind with more."""
        rows = lambda errs: [{"n": n, "error": e, "spread": 0.002, "rmse": e, "rmse_spread": 0.002, "outside": o, "outside_spread": 0.01}   # noqa: E731
                             for n, (e, o) in zip([16, 64, 256], errs)]
        base, cand = rows([(0.20, 0.9), (0.05, 0.6), (0.010, 0.5)]), rows([(0.10, 0.4), (0.06, 0.3), (0.030, 0.2)])
        return {"goal": "forecast fuel demand from price", "status": "done", "mode": "rehearsal", "reachable": False, "swarm": [],
                "cards": [], "unbuilt": [], "papers_read": 0,
                "brief": {"kind_name": "Tables and forecasts", "tasks": ["demand"], "task_names": ["demand against price"], "why": "x"},
                "rules": [{"id": "monotone", "name": "Monotonicity", "evidence": 0}],
                "tasks": [{"id": "demand", "name": "Demand against price", "holds": ["monotone"], "shape": [3], "sizes": [16, 64, 256],
                           "seeds": 2, "steps": 2700, "primary": "rmse", "winner": None,
                           "errors": [{"key": "rmse", "label": "error", "short": "rmse"},
                                      {"key": "outside", "label": "error outside the range it was trained on", "short": "Outside range"}],
                           "baseline": {"params": 4545, "kind": "MLP on the flattened input (tanh)", "rows": base, "ms": 0.2,
                                        "breaks": {"monotone": 0.21}},
                           "candidates": [{"rules": ["monotone"], "hidden": 32, "params": 1217, "rows": cand, "true": ["monotone"],
                                           "untrue": [], "ms": 0.3, "holds": {"monotone": 0.0}, **SA.compare(cand, base)}]}]}

    @check("every finding is explained: what it means and how far it reaches, for a win, a loss and a run that read nothing")
    def _():
        import superexplain as SX
        d = KEEP["won"]
        ex = d["explain"]                                            # the run explained itself before it reported
        assert [s["id"] for s in d["stages"]][-2:] == ["explain", "report"]
        ids = [f["id"] for f in ex["findings"]]
        for want in ("fit", "reading", "two_body.result", "two_body.rules", "two_body.size", "two_body.breaks", "unbuilt"):
            assert want in ids, (want, ids)
        assert all(len(f[k]) > 30 for f in ex["findings"] for k in ("finding", "means", "scope")), "an empty explanation"
        by = {f["id"]: f for f in ex["findings"]}
        assert "Nothing was trained on data from the goal itself" in by["fit"]["scope"] and "two bound bodies" in by["fit"]["finding"]
        assert "does not show that the rule is true of your problem" in by["reading"]["scope"]
        assert "this one mock task" in by["two_body.result"]["scope"] and "not one tuned" in by["two_body.result"]["scope"]
        assert "derivative" in by["two_body.size"]["means"]                                  # why an energy network runs slower
        assert "by construction" in by["two_body.breaks"]["scope"] or "how it is wired" in by["two_body.breaks"]["scope"]
        assert "does not support" in ex["bottom"] and "never trained on" in ex["bottom"]
        # a loss: the rule is true of the task and still did not pay; the explanation must not call the rule false
        lost = SX.explain(_lost_run())
        by = {f["id"]: f for f in lost["findings"]}
        r = by["demand.result"]
        assert r["tone"] == "caution" and "ahead at 16" in r["finding"] and "behind at 64 and 256" in r["finding"], r["finding"]
        assert "stopped helping once there were enough" in r["means"] and "does not show the rule is false" in r["scope"]
        assert "guarantee" in r["scope"] and "demand.size" not in by and "demand.data" not in by     # nothing to slim, no saving claimed
        assert by["demand.outside"]["tone"] == "good" and "where there were no examples" in by["demand.outside"]["means"]
        assert "No paper was read" in by["reading"]["finding"] and "would be the same numbers" in by["reading"]["scope"]
        assert "did not beat a conventional network" in lost["bottom"]
        # a run that ended before measuring says so, rather than explaining numbers it does not have
        cut = _lost_run()
        cut["tasks"][0]["candidates"] = []
        assert "not measured" in {f["id"]: f for f in SX.explain(cut)["findings"]}["demand.result"]["finding"]
        assert "nothing to conclude" in SX.explain({**cut, "tasks": []})["bottom"]

    @check("a model may reword what a finding means, but not add a number, and old runs are explained when opened")
    def _():
        import superexplain as SX
        d = json.loads(json.dumps(KEEP["won"]))
        draft = {f["id"]: f for f in SX.findings(d)}

        def ask(system, prompt, max_tokens=0):
            assert "never invent" in system.lower() and d["goal"] in prompt and "two_body.result" in prompt
            return {"findings": [{"id": "two_body.result", "means": "For bodies joined by a spring, wiring in the conservation laws made the model clearly more accurate.",
                                  "scope": "This was 99.7% accurate on real spacecraft data."},
                                 {"id": "fit", "means": "short", "scope": "Only the mock task was trained on; your own system was not, so this is a pointer and not a result."},
                                 {"id": "made-up", "means": "x" * 50, "scope": "y" * 50}],
                    "fit": "near", "fit_why": "The goal is this task.", "bottom": "You can take the direction from this run, and you cannot take a finished model; train on your own data next."}
        ex = SX.explain(d, ask=ask)
        by = {f["id"]: f for f in ex["findings"]}
        assert ex["by"] == "model" and ex["fit"] == "near" and ex["bottom"].startswith("You can take")
        assert by["two_body.result"]["means"].startswith("For bodies joined")                 # reworded
        assert by["two_body.result"]["scope"] == draft["two_body.result"]["scope"]            # 99.7 is in no finding: refused
        assert by["two_body.result"]["finding"] == draft["two_body.result"]["finding"]        # the numbers are never the model's
        assert by["fit"]["means"] == draft["fit"]["means"] and by["fit"]["scope"].startswith("Only the mock task")
        assert "made-up" not in by
        assert SX.explain(d, ask=lambda *a: None)["by"] == "numbers" and SX.explain(d, ask=lambda *a: "nonsense")["by"] == "numbers"
        # a run saved before it could explain itself gets an explanation on reading, and Explain again keeps one
        home = KEEP["home"]
        real = (SA.home, SA._live, SA._ask)
        SA.home, SA._live, SA._ask = (lambda: home), (lambda: True), ask
        try:
            old = {k: v for k, v in d.items() if k != "explain"}
            old["id"] = "old-run"
            (home / "old-run.json").write_text(json.dumps(old))
            assert SA.read("old-run")["explain"]["by"] == "numbers"
            assert "explain" not in json.loads((home / "old-run.json").read_text())           # reading does not rewrite the file
            again = SA.reexplain("old-run")
            assert again["explain"]["by"] == "model" and json.loads((home / "old-run.json").read_text())["explain"]["fit"] == "near"
            import main
            from fastapi.testclient import TestClient
            c = TestClient(main.app)
            assert c.post("/api/superagent/old-run/explain").json()["explain"]["by"] == "model"
            assert c.post("/api/superagent/nope/explain").status_code == 404
        finally:
            SA.home, SA._live, SA._ask = real
        page = (ROOT / "frontend" / "index.html").read_text()
        assert "function saExplain" in page and "How far it reaches" in page and 'id="saExplain"' in page

    @check("the page, the rail and the server know the super agent")
    def _():
        page = (ROOT / "frontend" / "index.html").read_text()
        assert 'data-page="pageSuper"' in page and 'id="superMain"' in page and "pageSuper: renderSuperPage" in page
        assert 'pageSuper:"super"' in page and "async function renderSuperPage" in page
        import main
        from fastapi.testclient import TestClient
        home = Path(tempfile.mkdtemp())
        real = SA.home
        SA.home = lambda: home
        try:
            c = TestClient(main.app)
            o = c.get("/api/superagent").json()
            assert len(o["rules"]) == len(SA.RULES) and [k["id"] for k in o["kinds"]] == list(SA.KINDS) and o["runs"] == []
            assert c.post("/api/superagent", json={"goal": ""}).status_code == 400
            assert c.get("/api/superagent/nope").status_code == 404
            assert c.get("/api/superagent/nope/canvas").status_code == 404
        finally:
            SA.home = real
        specs = {s["name"] for s in __import__("layers").catalog()}
        assert {"RuleNet", "SeqNet"} <= specs


print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    for name, why in FAILED:
        print(f"  {name}: {why}")
    sys.exit(1)
