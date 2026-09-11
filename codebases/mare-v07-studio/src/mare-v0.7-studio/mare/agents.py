from __future__ import annotations

import json

from .contracts import (
    AssassinOutput,
    DirectorOutput,
    ExplorerOutput,
    QuestionGeneratorOutput,
    SynthesisOutput,
    VerificationPlan,
)
from .memory import ResearchMemory
from .models import Claim, ProblemSpec, ResearchState, ResearchTask
from .providers.base import ModelProvider


class Director:
    SYSTEM = """You are the research director. Decompose the problem into genuinely different research programs. Do not prove the theorem yourself. Prefer disagreement and falsifiable tasks over consensus."""

    def __init__(self, provider: ModelProvider):
        self.provider = provider

    async def plan(self, problem: ProblemSpec) -> DirectorOutput:
        return await self.provider.generate(
            system=self.SYSTEM,
            prompt=problem.model_dump_json(indent=2),
            schema=DirectorOutput,
            temperature=0.4,
        )


class Explorer:
    SYSTEM = """You are an independent mathematical explorer. Work only on the assigned task. Emit atomic claims with derivations, assumptions, explicit dependencies, and deterministic checks that could falsify them. Use relevant prior successes and failures but do not blindly trust them. Do not claim success for the overall problem unless the task logically warrants it."""

    def __init__(self, provider: ModelProvider, name: str, memory: ResearchMemory | None = None):
        self.provider = provider
        self.name = name
        self.memory = memory or ResearchMemory()

    async def investigate(self, problem: ProblemSpec, task: ResearchTask, state: ResearchState) -> ExplorerOutput:
        retrieved = self.memory.retrieve(state, task)
        prompt = {
            "problem": problem.model_dump(),
            "task": task.model_dump(),
            "relevant_prior_claims": [c.model_dump() for c in retrieved.claims],
            "relevant_failures": [f.model_dump() for f in retrieved.failures],
            "open_questions": [
                q.model_dump()
                for q in state.questions
                if q.status.value == "open" and (q.branch_id in {None, task.branch_id})
            ][:6],
        }
        return await self.provider.generate(
            system=self.SYSTEM,
            prompt=json.dumps(prompt, indent=2, default=str),
            schema=ExplorerOutput,
            temperature=0.7,
            max_tokens=task.token_budget,
        )


class Assassin:
    SYSTEM = """You are an adversarial mathematical reviewer. Your reward is for finding fatal defects, not for being agreeable. Attack hidden assumptions, circularity, illegal limits, sign mistakes, counterexamples, conservation-law conflicts, and mismatch with the original problem. Do not repair the proof."""

    def __init__(self, provider: ModelProvider, name: str = "assassin"):
        self.provider = provider
        self.name = name

    async def attack(self, problem: ProblemSpec, claim: Claim) -> AssassinOutput:
        prompt = json.dumps({"problem": problem.model_dump(), "claim": claim.model_dump()}, indent=2, default=str)
        return await self.provider.generate(
            system=self.SYSTEM,
            prompt=prompt,
            schema=AssassinOutput,
            temperature=0.3,
        )


class VerifierPlanner:
    SYSTEM = """You plan deterministic verification. Choose only checks relevant to the claim. Your output is a test plan, not a verdict; the tool runner decides the evidence result."""

    def __init__(self, provider: ModelProvider):
        self.provider = provider

    async def plan(self, claim: Claim) -> VerificationPlan:
        return await self.provider.generate(
            system=self.SYSTEM,
            prompt=claim.model_dump_json(indent=2),
            schema=VerificationPlan,
            temperature=0.0,
        )


class Synthesizer:
    SYSTEM = """You connect already-surviving research branches. Do not invent unsupported premises. New claims must explicitly derive from the supplied verified/surviving claims and should expose cross-branch dependencies."""

    def __init__(self, provider: ModelProvider):
        self.provider = provider

    async def synthesize(self, state: ResearchState) -> SynthesisOutput:
        claims = [
            c.model_dump()
            for c in state.claims
            if c.status.value in {"survived_review", "verified", "formalized"}
        ]
        failures = [f.model_dump() for f in state.failures[-10:]]
        return await self.provider.generate(
            system=self.SYSTEM,
            prompt=json.dumps(
                {"problem": state.problem.model_dump(), "claims": claims, "recent_failures": failures},
                indent=2,
                default=str,
            ),
            schema=SynthesisOutput,
            temperature=0.4,
        )


class QuestionGenerator:
    SYSTEM = """You are a research-question generator. Your job is not to solve the theorem. Generate a small number of high-value next questions that discriminate between hypotheses, resolve contradictions, close proof dependencies, or transfer a successful mechanism. Prefer questions whose answer would change the research direction. Flag a new branch only when the question represents a genuinely distinct mechanism."""

    def __init__(self, provider: ModelProvider):
        self.provider = provider

    async def generate(self, state: ResearchState) -> QuestionGeneratorOutput:
        payload = {
            "problem": state.problem.model_dump(),
            "branches": [b.model_dump() for b in state.branches],
            "verified_claims": [c.model_dump() for c in state.claims if c.status.value == "verified"],
            "rejected_claims": [c.model_dump() for c in state.claims if c.status.value == "rejected"][-8:],
            "failures": [f.model_dump() for f in state.failures[-8:]],
            "existing_questions": [q.model_dump() for q in state.questions if q.status.value == "open"],
        }
        return await self.provider.generate(
            system=self.SYSTEM,
            prompt=json.dumps(payload, indent=2, default=str),
            schema=QuestionGeneratorOutput,
            temperature=0.55,
        )
