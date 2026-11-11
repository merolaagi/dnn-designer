"""superexplain.py — what each finding of a super agent run means, and how far it reaches.

A report of numbers leaves two questions open for every line in it: what does this tell me, and what does
it not tell me. This reads a finished run (any run: a win, a loss, no winner, nothing read) and answers both
for each finding:

    finding   the measured fact, with its numbers, taken from the run and never reworded by a model
    means     what follows from it, in plain words
    scope     how far it reaches: the conditions it was measured under, and what it does not show

The findings and their numbers are worked out here from the run alone, so every run can be explained with
no model connected. With a model, it reads the same findings and rewrites `means` and `scope` for this
particular goal, and judges how close the mock tasks are to the goal; it is not allowed to add numbers.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

WHY_SLOW = {"energy": "a motion read off an energy needs a derivative on every pass",
            "permutation": "every pair of rows is compared",
            "reflection": "the network is run on the input and on its mirror image",
            "fastweights": "the memory is updated step by step along the list",
            "additive": "each input has its own small network"}


def _n(v) -> str:
    if v is None:
        return "—"
    v = float(v)
    if v == 0:
        return "0"
    a = abs(v)
    return f"{v:.0f}" if a >= 100 else f"{v:.2f}" if a >= 1 else f"{v:.3g}" if a >= 0.001 else f"{v:.1e}"


def _join(items: List[str]) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1] if items else ""


def _rules(c) -> str:
    return " + ".join(c["rules"])


def _same(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> bool:
    """Two networks cannot be told apart when, at every size, their errors differ by less than their spreads."""
    return all(abs(x["error"] - y["error"]) <= x["spread"] + y["spread"] + 1e-12 for x, y in zip(a, b))


def _conditions(T) -> str:
    return (f"{T['sizes'][0]} to {T['sizes'][-1]} training examples, {T['seeds']} seeds and "
            f"{T['steps']:,} training steps")


def _task_findings(d, T, names) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    tid, about = T["id"], T["name"]
    cands, base = T.get("candidates") or [], T.get("baseline") or {}
    rows_b = base.get("rows") or []
    if not cands or not rows_b:
        return [{"id": f"{tid}.result", "about": about, "tone": "caution",
                 "finding": "This task was not measured: the run ended before its training finished.",
                 "means": "There is no result for it either way.",
                 "scope": "Nothing can be read from this task. Run it again to get its numbers."}]
    sizes = T["sizes"]
    win = cands[T["winner"]] if T.get("winner") is not None else None
    best = win or max(cands, key=lambda c: c["gain"])
    errs = lambda rows: ", ".join(_n(r["error"]) for r in rows)      # noqa: E731
    plain = "a GRU" if base.get("kind") == "GRU" else "a plain MLP"

    # ---- the headline result ----------------------------------------------------------------
    if win:
        ratios = win.get("ratios") or []
        trend = ""
        if len(ratios) > 1:
            first, last = ratios[0], ratios[-1]
            trend = (f" The gap grew with data, from {first}× at {sizes[0]} examples to {last}× at {sizes[-1]}."
                     if last > first * 1.25 else
                     f" The gap narrowed with data, from {first}× at {sizes[0]} examples to {last}× at {sizes[-1]}, "
                     "as it usually does: with enough data a plain network can learn what the rule gives for free."
                     if first > last * 1.25 else
                     f" The gap held steady across the sizes ({first}× to {last}×).")
        out.append({"id": f"{tid}.result", "about": about, "tone": "good",
                    "finding": f"The network built on {_rules(win)} had {win['gain']}× lower error than the conventional one, on "
                               f"average over the sizes (errors {errs(win['rows'])} against {errs(rows_b)} at n = {_join(sizes)}).",
                    "means": f"On this task, building the rule{'s' if len(win['rules']) > 1 else ''} into the network gave a real gain in "
                             f"accuracy: it was ahead by more than the run-to-run variation at {_join(win['wins'])} examples and behind at none.",
                    "scope": f"Measured on this one mock task, with {_conditions(T)}. The conventional network is {plain} of "
                             f"{base['params']:,} weights with default settings, not one tuned for the task.{trend}"})
        if win.get("reaches_at") and win["reaches_at"] < sizes[-1]:
            out.append({"id": f"{tid}.data", "about": about, "tone": "good",
                        "finding": f"With {win['reaches_at']} examples it was already as accurate as the conventional network with {sizes[-1]}.",
                        "means": f"It needs roughly {win['data_saving']:g}× less data to reach that accuracy. This is the main thing a "
                                 "built-in rule buys: what the rule states does not have to be learned from examples.",
                        "scope": f"{sizes[-1]} examples was the most that was tried, and the comparison is at one accuracy level. "
                                 "The saving is on data generated without measurement error; noisy real data would shrink it."})
    else:
        behind = best["gain"] < 0.95
        early = bool(best["wins"]) and bool(best["losses"]) and max(best["wins"]) < min(best["losses"])
        means = ("The rule helped when examples were scarce and stopped helping once there were enough of them: a plain network "
                 "with enough data learned what the rule would have given it." if early else
                 "On this task the rules cost accuracy. A constrained network is harder to train and less free to fit, and here "
                 "that cost outweighed what the rule gave." if behind else
                 "Building the rules in made no difference that can be told from run-to-run variation.")
        true = [names.get(r, r) for r in best["rules"] if r in T["holds"]]
        out.append({"id": f"{tid}.result", "about": about, "tone": "bad" if behind and not early else "caution",
                    "finding": f"No rule-built network stayed ahead of the conventional one. The closest, {_rules(best)}, averaged "
                               f"{best['gain']}× (errors {errs(best['rows'])} against {errs(rows_b)} at n = {_join(sizes)})"
                               + (f"; ahead at {_join(best['wins'])}" if best["wins"] else "")
                               + (f", behind at {_join(best['losses'])}" if best["losses"] else "") + ".",
                    "means": means,
                    "scope": (f"This does not show the rule is false: {_join(true)} {'is' if len(true) == 1 else 'are'} true of this "
                              "mock task by the way its data are made. " if true else "")
                             + f"It shows that with {_conditions(T)}, building it in did not pay in accuracy. What a rule still "
                               "gives without an accuracy win is the guarantee: the network cannot break it, on any input."})

    # ---- which rule did the work --------------------------------------------------------------
    core = "energy" if T.get("shape") and all("energy" in c["rules"] for c in cands) else None
    single = [c for c in cands if len([r for r in c["rules"] if r != core]) == 1 and not c["untrue"]]
    combos = [c for c in cands if len([r for r in c["rules"] if r != core]) > 1 and not c["untrue"]]
    untrue = [c for c in cands if c["untrue"]]
    parts, means, scope = [], [], []
    if single:
        parts.append("Alone: " + ", ".join(f"{_rules(c)} {c['gain']}×" for c in single) + ".")
    if combos:
        parts.append("Together: " + ", ".join(f"{_rules(c)} {c['gain']}×" for c in combos) + ".")
    if single and combos:
        top_s, top_c = max(c["gain"] for c in single), max(c["gain"] for c in combos)
        if top_c > top_s * 1.3:
            means.append("The rules compound: together they gave more than any one of them alone, because each removes a "
                         "different thing the network would otherwise have to learn.")
        elif top_s > top_c * 1.3:
            means.append("One rule did the work, and adding others on top of it did not help.")
        weak = [c for c in single if 0.9 <= c["gain"] <= 1.15]
        if weak and top_c > 1.5:
            means.append(f"{_join([_rules(c) for c in weak]).capitalize()} alone made almost no difference, yet helped in "
                         "combination: a rule can be true and still be worth little on its own.")
    if win:
        others = [c for c in cands if c is not win and not c["losses"] and _same(c["rows"], win["rows"])]
        if others:
            extra = sorted(set(win["rules"]) - set(others[0]["rules"])) or sorted(set(others[0]["rules"]) - set(win["rules"]))
            parts.append(f"The winner and {_rules(others[0])} differ by less than the run-to-run variation at every size.")
            means.append("The winner is not measurably better than that one. "
                         + (f"The extra rule{'s' if len(extra) > 1 else ''} ({_join([names.get(r, r) for r in extra])}) made no "
                            "difference that this run can detect; they are in the winner because they are true and cost nothing here."
                            if extra else ""))
    if untrue:
        parts.append("Built on a rule that is not true of this task: " + ", ".join(f"{_rules(c)} {c['gain']}×" for c in untrue) + ".")
        worse = [c for c in untrue if c["gain"] < 1]
        if worse:
            means.append("Networks built on a rule that is false for the task came out worse than the conventional one. The "
                         "comparison can tell a wrong rule from a right one: a false rule shows as a loss, not as a quiet win.")
        scope.append("The run knows which rules are true only because it made the data. On your own data nobody marks them: "
                     "a rule-built network losing there is the sign that its rule does not hold.")
    if parts:
        scope.append("Which rule matters most is a fact about this task, not about the rule in general.")
        out.append({"id": f"{tid}.rules", "about": about, "tone": "neutral", "finding": " ".join(parts),
                    "means": " ".join(means) or "Each line is the same comparison with a different set of rules built in, so the "
                                                "differences between lines show what each rule contributed.",
                    "scope": " ".join(scope)})

    # ---- size and speed -----------------------------------------------------------------------
    slim = T.get("slim")
    if win and slim:
        full = win["rows"][-1]["error"]
        cost = slim["error"] / max(full, 1e-12)
        faster = slim.get("faster") or 1.0
        why = next((WHY_SLOW[r] for r in ("energy", "permutation", "reflection", "fastweights", "additive") if r in win["rules"]), "")
        speed = (f"It is {faster}× faster to run." if faster >= 1.15 else
                 f"It is {round(1 / max(faster, 1e-9), 1)}× slower to run" + (f", because {why}." if why else ".")
                 if faster <= 0.87 else "It runs about as fast.")
        out.append({"id": f"{tid}.size", "about": about, "tone": "good" if faster >= 0.87 else "caution",
                    "finding": f"Slimmed to {slim['params']:,} weights against the conventional network's {base['params']:,} "
                               f"({slim['smaller']}× smaller). One pass over 1,000 rows takes {_n(slim['ms'])} ms against {_n(base.get('ms'))} ms.",
                    "means": f"The rule lets a far smaller network do the job, which makes it cheaper to store and to train. {speed}",
                    "scope": (f"Slimming gave up accuracy: the slimmed network's error is {_n(slim['error'])}, {cost:.1f}× the full-size "
                              f"winner's {_n(full)}, though still below the conventional network's {_n(rows_b[-1]['error'])}. If accuracy "
                              f"matters more than size, use hidden width {win['hidden']} ({win['params']:,} weights). "
                              if cost > 1.3 else
                              f"The slimmed network kept the full-size winner's accuracy ({_n(slim['error'])} against {_n(full)}). ")
                             + "\"Smaller\" counts weights only. The times are for this machine's processor and are tiny for both; "
                               "they matter only if the model runs many millions of times."})

    # ---- second measures: outside the range, conserved quantities --------------------------------
    for e in T.get("errors") or []:
        key = e["key"]
        if key == T["primary"] or key not in (rows_b[-1] if rows_b else {}):
            continue
        b, c = rows_b[-1].get(key), best["rows"][-1].get(key)
        if b is None or c is None:
            continue
        ratio = b / max(c, 1e-12)
        who = "the winner" if win else f"the closest network ({_rules(best)})"
        better = ratio > 1.25
        out.append({"id": f"{tid}.{key}", "about": about, "tone": "good" if better else "bad" if ratio < 0.8 else "neutral",
                    "finding": f"{e.get('short') or key}: {_n(c)} for {who} against {_n(b)} for the conventional network "
                               f"({e['label']}; at {sizes[-1]} examples).",
                    "means": (EXTRA_MEANS.get(key, "On this second measure the rule-built network is {x}× better.") if better else
                              "On this second measure the rule-built network is no better." if ratio >= 0.8 else
                              "On this second measure the rule-built network is worse.").format(x=_n(ratio)),
                    "scope": EXTRA_SCOPE.get(key, "Measured at the largest size only, on one mock task.")})

    # ---- what the conventional network failed to learn --------------------------------------------
    breaks = [(r, v) for r, v in (base.get("breaks") or {}).items() if v is not None and r in T["holds"]]
    broke = [(r, v) for r, v in breaks if v > 0.01]
    if broke:
        out.append({"id": f"{tid}.breaks", "about": about, "tone": "neutral",
                    "finding": f"After training on {sizes[-1]} examples the conventional network still broke "
                               + _join([f"{names.get(r, r).lower()} by {_n(v)}" for r, v in broke])
                               + " (0 would mean it holds). The rule-built networks hold their rules at 0.",
                    "means": "The conventional network did not pick the rule up from the data, even though every example obeyed it. "
                             "A rule that is built in is kept exactly; a rule that has to be learned is only approximated.",
                    "scope": "For the rule-built network the 0 is a property of how it is wired, true before any training and on any "
                             "input; it is not a training result. The conventional network's figure was measured on inputs like the "
                             "ones it trained on, and is usually worse elsewhere."})
    return out


EXTRA_MEANS = {
    "outside": "Outside the range it was trained on, the rule-built network is {x}× more accurate: a rule keeps holding where there were no examples.",
    "rollout": "Followed forward step by step, its predicted path stays {x}× closer to the true one.",
    "energy": "Its predicted motion keeps the energy {x}× better, so long predictions do not drift away.",
    "angular": "Its predicted motion keeps the angular momentum {x}× better. Nobody trained it to: a rotation-invariant energy conserves it by Noether's theorem.",
    "momentum": "Its predicted motion keeps the total momentum {x}× better. Nobody trained it to: a translation-invariant energy conserves it by Noether's theorem.",
}
EXTRA_SCOPE = {
    "outside": "Inputs up to 25% beyond the training range, at the largest size only. Further out, every network's error grows.",
    "rollout": "One starting state, followed for 10 time units. Other starting states can behave differently.",
    "energy": "One starting state, followed for 10 time units, measured against the true system's energy.",
    "angular": "One starting state, followed for 10 time units. With the rotation rule built in this would be near 0 by construction.",
    "momentum": "One starting state, followed for 10 time units. With the translation rule built in this is 0 by construction.",
}


def findings(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every finding of a run, each with what it means and how far it reaches, worked out from the run alone."""
    import superagent as SA

    out: List[Dict[str, Any]] = []
    names = {rid: r["name"] for rid, r in SA.RULES.items()}
    goal = (d.get("goal") or "").strip()
    short_goal = goal if len(goal) <= 110 else goal[:110].rstrip() + "…"
    br = d.get("brief") or {}
    tasks = d.get("tasks") or []

    # ---- the goal and what stood in for it ----------------------------------------------------
    if br:
        SA._load_words()
        hits = sorted({m.group(0).lower() for t in br.get("tasks") or [] for w in SA.WORDS_OF.get(t, [])
                       for m in [re.search(r"\w*(?:" + w + r")\w*", goal, re.I)] if m})
        loose = ("None of the goal's own words name these tasks, so treat the match as loose. " if not hits else
                 f"Only one of the goal's own words points at these tasks (“{hits[0]}”), so treat the match as loose. "
                 if len(hits) == 1 else
                 f"The goal's own words that point at these tasks: {_join(hits[:5])}. ")
        radar = ("The goal was a headline picked from Radar, not a problem you described, which makes the match looser still. "
                 if (d.get("source") or {}).get("kind") == "radar" else "")
        out.append({"id": "fit", "about": "The goal and what stood in for it", "tone": "caution",
                    "finding": f"The goal was read as {str(br.get('kind_name') or '').lower()} and tested on "
                               f"{len(br.get('tasks') or [])} mock task{'s' if len(br.get('tasks') or []) != 1 else ''}: "
                               f"{_join(br.get('task_names') or [])}.",
                    "means": "A mock task is a small problem with the same kind of structure as the goal, whose right answer is known "
                             "exactly, so error can be measured. " + (str(br.get("why") or "")),
                    "scope": f"Nothing was trained on data from the goal itself. {loose}{radar}Every result below is about whether a rule "
                             f"helps on problems shaped like the mock tasks; none of them shows that “{short_goal}” is solved."})

    # ---- reading ------------------------------------------------------------------------------
    rules = d.get("rules") or []
    cards = d.get("cards") or []
    if d.get("swarm") is not None and rules:
        found = [r["name"] for r in rules if r.get("evidence")]
        lib = [r["name"] for r in rules if not r.get("evidence")]
        empty = sum(1 for a in d.get("swarm") or [] if a.get("status") == "done" and a.get("papers") and not a.get("cards"))
        if d.get("reachable") is False:
            out.append({"id": "reading", "about": "What was read", "tone": "caution",
                        "finding": "No paper was read: the literature search could not be reached from this machine.",
                        "means": f"The rules that were tried ({_join([r['name'] for r in rules])}) came from this lab's own library, "
                                 "which lists well-known results with their sources.",
                        "scope": "The reading step contributed nothing to this run. The measurements do not depend on it: they would "
                                 "be the same numbers with or without papers. What is missing is any rule the papers might have added."})
        else:
            by = ("The assistants matched rules by keywords, not by reading for meaning, so a card can be a passing mention. "
                  if d.get("mode") != "live" else "")
            out.append({"id": "reading", "about": "What was read", "tone": "neutral",
                        "finding": f"{d.get('papers_read', 0)} papers were looked at and {len(cards)} rule card{'s' if len(cards) != 1 else ''} "
                                   f"came back, covering {len(found)} of the {len(rules)} rules for this kind of problem"
                                   + (f" ({_join(found)})." if found else "."),
                        "means": "A rule card is a sentence in a paper that states a rule this lab can build into a network. "
                                 + (f"{_join(lib)} {'was' if len(lib) == 1 else 'were'} not found in the reading and "
                                    f"{'was' if len(lib) == 1 else 'were'} tried anyway from the lab's library. " if lib else "")
                                 + (f"{empty} of the assistants found papers on the subject that state no rule of this kind, so they "
                                    "added nothing to the designs." if empty else ""),
                        "scope": f"{by}A card shows that a paper states the rule. It does not show that the rule is true of your "
                                 "problem: the training comparison is what tests that. Most papers were read by title and abstract only."})

    # ---- each task ----------------------------------------------------------------------------
    for T in tasks:
        out += _task_findings(d, T, names)

    # ---- rules read and not built -------------------------------------------------------------
    unbuilt = d.get("unbuilt") or []
    if unbuilt:
        out.append({"id": "unbuilt", "about": "Read, and not built", "tone": "neutral",
                    "finding": f"{len(unbuilt)} rule{'s were' if len(unbuilt) != 1 else ' was'} read that cannot be built in yet: "
                               + _join([str(c.get("name") or "a rule") for c in unbuilt[:6]]) + ".",
                    "means": f"The papers state {'it' if len(unbuilt) == 1 else 'them'}, and there is no construction here yet that "
                             f"would make a network hold {'it' if len(unbuilt) == 1 else 'them'} for any weights.",
                    "scope": "Nothing in this report's numbers used "
                             + ("it. It is what to build next, and it may be the rule that matters most for the goal."
                                if len(unbuilt) == 1 else
                                "them. They are the list of what to build next, and they may be the rules that matter most for the goal.")})
    return out


def bottom_line(d: Dict[str, Any]) -> str:
    tasks = d.get("tasks") or []
    won = [T for T in tasks if T.get("winner") is not None]
    goal = (d.get("goal") or "").strip()
    short_goal = goal if len(goal) <= 90 else goal[:90].rstrip() + "…"
    if not tasks:
        return "The run ended before anything was measured, so there is nothing to conclude from it."
    if won:
        supports = (f"What this run supports: on {_join([T['name'].lower() for T in won])}, a network with the right rules built "
                    "in is more accurate, needs less data and is far smaller than a conventional one.")
    else:
        supports = "What this run supports: on these mock tasks, building the rules in did not beat a conventional network on accuracy."
    return (f"{supports} What it does not support: any claim about “{short_goal}” itself, which was never trained on. To test "
            "the goal, open the winning design on the canvas and train it on data from the real problem: there, a rule that is "
            "not true of the data will show up as a loss.")


EXPLAIN = """You explain the results of an experiment to its owner, who is a capable non-specialist. The experiment built small
neural networks with known rules wired into their structure, trained them on MOCK tasks (data generated from equations),
and compared them with a conventional network.

You are given the owner's goal, and a list of findings. Each finding has an id, the measured fact ("finding"), and a
draft of what it means ("means") and how far it reaches ("scope").

Rewrite "means" and "scope" for each finding so they speak to THIS goal. Rules you must follow:
- Plain words. Two or three sentences each. No jargon without saying what it is.
- Use only numbers that appear in the finding. Never invent or recompute a number.
- "scope" must say what the finding does NOT show. Never say or imply that the goal itself was solved or tested:
  only mock tasks were trained on.
- Keep every caution in the draft; you may sharpen it.
Also judge how close the mock tasks are to the goal: "near" (the goal is this kind of problem), "loose" (shares a
structure, but is a different problem), or "far" (little to do with it), with one sentence saying why. And write a
bottom line of two or three sentences: what the owner can and cannot take from this run, and the single most useful
next step toward the goal.

Answer with JSON only:
{"findings": [{"id": "<id>", "means": "...", "scope": "..."}], "fit": "near|loose|far", "fit_why": "...", "bottom": "..."}"""


def explain(d: Dict[str, Any], ask: Optional[Callable] = None) -> Dict[str, Any]:
    """The explanation of a run. `ask(system, prompt, max_tokens)` is a model call returning parsed JSON or None;
    without it (or when it fails) the explanation is the one worked out from the numbers."""
    items = findings(d)
    out = {"by": "numbers", "findings": items, "bottom": bottom_line(d), "fit": "", "fit_why": ""}
    if not ask or not items:
        return out
    listing = json.dumps([{k: f[k] for k in ("id", "finding", "means", "scope")} for f in items], ensure_ascii=False)
    got = ask(EXPLAIN, f"Goal: {d.get('goal', '')}\n\nFindings:\n{listing}", 3500)
    if not isinstance(got, dict):
        return out
    by_id = {str(x.get("id")): x for x in (got.get("findings") or []) if isinstance(x, dict)}
    changed = 0
    for f in items:
        new = by_id.get(f["id"]) or {}
        for key in ("means", "scope"):
            text = re.sub(r"\s+", " ", str(new.get(key) or "")).strip()
            if 20 <= len(text) <= 900 and _numbers_ok(text, f):
                f[key] = text
                changed += 1
    fit = str(got.get("fit") or "").strip().lower()
    if fit in ("near", "loose", "far"):
        out["fit"], out["fit_why"] = fit, re.sub(r"\s+", " ", str(got.get("fit_why") or ""))[:400]
    bottom = re.sub(r"\s+", " ", str(got.get("bottom") or "")).strip()
    if 40 <= len(bottom) <= 900:
        out["bottom"] = bottom
    if changed:
        out["by"] = "model"
    return out


def _numbers_ok(text: str, f: Dict[str, Any]) -> bool:
    """A rewritten sentence may only use numbers that the finding or its draft already contains."""
    known = set(re.findall(r"\d+(?:\.\d+)?", " ".join(str(f[k]) for k in ("finding", "means", "scope")).replace(",", "")))
    for num in re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")):
        if num not in known and num.rstrip("0").rstrip(".") not in {k.rstrip("0").rstrip(".") for k in known}:
            return False
    return True
