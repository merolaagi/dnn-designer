"""Opt-in recorder and frozen-model adapter. This module never verifies a claim."""

import hashlib
import pickle
from pathlib import Path

from mare.learned_policy import ResearchPolicy
from .features import graph_record


def metrics(state, branch_id):
    claims = [c for c in state.claims if c.branch_id == branch_id]
    ids = {c.id for c in claims}
    return dict(
        verified=sum(c.status.value in {"verified", "formalized"} for c in claims),
        counterexamples=sum(
            e.claim_id in ids
            and e.evidence_type == "counterexample_search"
            and e.result.value == "contradicts"
            and e.reproducible
            for e in state.evidence
        ),
        answered=sum(q.branch_id == branch_id and q.status.value == "answered" for q in state.questions),
        work=next((b.compute_spent for b in state.branches if b.id == branch_id), 0),
    )


class GraphResearchPolicy(ResearchPolicy):
    def __init__(self, *, checkpoint=None, mode="shadow", source="mock", enabled=True):
        super().__init__(enabled)
        if mode not in {"shadow", "blend"}:
            raise ValueError("mode must be shadow or blend")
        if source not in {"mock", "real"}:
            raise ValueError("source must be mock or real")
        self.mode, self.source, self.model = mode, source, None
        self.status = dict(mode=mode, active=False, reason="Recording only; no checkpoint", checkpoint=None)
        if checkpoint:
            try:
                import torch

                torch.set_num_threads(1)
                from .network import load_checkpoint

                path = Path(checkpoint)
                if path.stat().st_size > 32 * 1024 * 1024:
                    raise ValueError("Checkpoint exceeds 32 MiB")
                model, metadata = load_checkpoint(path)
                if not metadata.get("trained_examples"):
                    raise ValueError("Checkpoint has no training examples")
                if mode == "blend" and metadata.get("source") != "real":
                    raise ValueError("Synthetic/mock checkpoints are shadow-only")
                self.model = model
                self.status["architecture"] = model.config
                self.status.update(
                    active=True,
                    reason="Frozen graph model loaded",
                    checkpoint=hashlib.sha256(path.read_bytes()).hexdigest(),
                    metadata=metadata,
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
                ImportError,
                KeyError,
                EOFError,
                pickle.UnpicklingError,
                TypeError,
            ) as exc:
                self.status["reason"] = f"Fallback: {type(exc).__name__}: {exc}"

    def _predict(self, state):
        data = state.planner_data
        pinned = data.get("checkpoint")
        current = self.status["checkpoint"]
        if pinned and pinned != current:
            data["status"] = dict(
                mode=self.mode,
                active=False,
                reason="Checkpoint changed on resume; falling back",
                checkpoint=pinned,
            )
            return {}
        data["status"] = self.status.copy()
        if self.model is None or not self.enabled or not state.policy.enabled:
            return {}
        if not pinned:
            data["checkpoint"] = current
        try:
            import torch
            from .network import collate

            record = graph_record(state)
            with torch.inference_mode():
                values = self.model(collate([record]))
            if not torch.isfinite(values).all():
                raise ValueError("Nonfinite predictions")
            result = {key: [float(v) for v in row] for key, row in zip(record["candidate_ids"], values)}
            data["predictions"] = result
            return result
        except (ValueError, RuntimeError) as exc:
            data["status"] = dict(self.status, active=False, reason=f"Fallback: {exc}")
            return {}

    def predict_branch(self, state, branch):
        predictions = self._predict(state)
        if self.mode == "blend" and branch.id in predictions:
            utility, cost, _ = predictions[branch.id]
            return utility / (1 + cost), min(0.25, max(0, state.policy.max_neural_blend))
        return super().predict_branch(state, branch)

    def score_questions(self, state):
        super().score_questions(state)
        predictions = self._predict(state)
        if self.mode == "blend":
            for q in state.questions:
                if q.id in predictions:
                    utility, cost, _ = predictions[q.id]
                    q.policy_score = utility / (1 + cost)
                    q.policy_blend = min(0.25, max(0, state.policy.max_neural_blend))

    def begin_round(self, state):
        super().begin_round(state)
        traces = state.planner_data.setdefault("traces", [])
        if any(r["round"] == state.current_round for r in traces):
            return
        try:
            record = graph_record(state)
        except ValueError as exc:
            state.planner_data["status"] = dict(self.status, active=False, reason=str(exc))
            return
        tasks = [t for t in state.tasks if t.round_no == state.current_round]
        affordable = []
        calls = state.budget.model_calls_used
        tokens = state.budget.reserved_output_tokens_used
        for task in tasks:
            if (
                calls + 1 <= state.budget.max_model_calls
                and tokens + task.token_budget <= state.budget.max_reserved_output_tokens
            ):
                affordable.append(task)
                calls += 1
                tokens += task.token_budget
        tasks = affordable
        cost_units = {}
        for task in tasks:
            for key in (task.branch_id, task.question_id):
                if key:
                    cost_units[key] = cost_units.get(key, 0.0) + task.token_budget / 4000.0
        selected = {t.branch_id for t in tasks} | {t.question_id for t in tasks if t.question_id}
        record.update(
            source=self.source,
            cost_units=cost_units,
            cost_basis="Reserved explorer output tokens / 4000; excludes verifier and synthesis costs",
            selection="deterministic scheduled set; conditional inclusion, not randomized propensity",
            selected=[i in selected for i in record["candidate_ids"]],
            inclusion_probability=[float(i in selected) for i in record["candidate_ids"]],
            before={b.id: metrics(state, b.id) for b in state.branches},
            targets=[[None] * 3 for _ in record["candidates"]],
        )
        traces.append(record)
        del traces[:-100]
        state.planner_data["status"] = self.status.copy()

    def learn_round(self, state):
        result = super().learn_round(state)
        traces = state.planner_data.get("traces", [])
        record = next((r for r in traces if r["round"] == state.current_round), None)
        if record is None or record.get("complete"):
            return result
        question_map = {q.id: q for q in state.questions}
        for i, key in enumerate(record["candidate_ids"]):
            if not record["selected"][i]:
                continue  # unobserved alternatives are not negative labels
            question = question_map.get(key)
            branch_id = question.branch_id if question else key
            before = record["before"].get(branch_id)
            if before is None:
                continue
            after = metrics(state, branch_id)
            gain = {k: max(0, after[k] - before[k]) for k in before}
            if question:
                resolved = float(question.status.value == "answered")
                answers = set(question.answered_by_claim_ids)
                utility = float(
                    any(
                        c.id in answers and c.status.value in {"verified", "formalized"} for c in state.claims
                    )
                )
                # A reproducible disproof is useful even when no answer is accepted.
                counter = any(
                    e.claim_id in answers
                    and e.evidence_type == "counterexample_search"
                    and e.result.value == "contradicts"
                    and e.reproducible
                    for e in state.evidence
                )
                utility = max(utility, 0.7 * counter)
                units = record["cost_units"][key]
                cost = units / (1 + units)
            else:
                utility = min(
                    1, 0.55 * gain["verified"] + 0.35 * gain["counterexamples"] + 0.1 * gain["answered"]
                )
                resolved = float(gain["answered"] > 0)
                units = record["cost_units"][key]
                cost = units / (1 + units)
            record["targets"][i] = [utility, cost, resolved]
        record["complete"] = True
        record.pop("before", None)
        return result
