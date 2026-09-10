from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from mare.contracts import (
    AssassinOutput,
    BranchPlan,
    DirectorOutput,
    ExplorerOutput,
    QuestionGeneratorOutput,
    ResearchQuestionDraft,
    SynthesisOutput,
    TaskPlan,
    VerificationPlan,
)
from mare.models import ClaimDraft, ClaimScope

T = TypeVar("T", bound=BaseModel)


class MockResearchProvider:
    """Deterministic research organization for architecture tests.

    It intentionally creates one false branch, then uses the failure to spawn a
    new reconciliation branch in round two. This lets tests exercise learning,
    question generation, branch spawning, and dependency-aware synthesis.
    """

    name = "mock"
    model = "deterministic-burgers-fixture-v3"

    async def generate(
        self,
        *,
        system: str,
        prompt: str,
        schema: type[T],
        temperature: float = 0.2,
        max_tokens: int = 4000,
    ) -> T:
        if schema is DirectorOutput:
            return DirectorOutput(
                branches=[
                    BranchPlan(
                        title="Characteristics",
                        research_question="What do characteristics imply about the solution map?",
                        strategy="derive characteristic flow and loss of invertibility",
                        novelty=0.45,
                    ),
                    BranchPlan(
                        title="Gradient dynamics",
                        research_question="How does u_x evolve along a characteristic?",
                        strategy="differentiate the PDE and derive an ODE for the gradient",
                        novelty=0.55,
                    ),
                    BranchPlan(
                        title="Energy mechanism",
                        research_question="Can increasing L2 energy itself force blow-up?",
                        strategy="look for a global energy-growth mechanism",
                        novelty=0.50,
                    ),
                ],
                tasks=[
                    TaskPlan(branch_title="Characteristics", title="Derive characteristic map", objective="Derive x(t) and u along characteristics."),
                    TaskPlan(branch_title="Gradient dynamics", title="Derive gradient ODE", objective="Derive and solve the ODE for w=u_x along characteristics."),
                    TaskPlan(branch_title="Energy mechanism", title="Test energy-growth hypothesis", objective="Determine whether L2 energy growth explains gradient blow-up."),
                ],
                rationale="Probe three mechanistically distinct explanations before converging.",
            )  # type: ignore[return-value]

        if schema is ExplorerOutput:
            low = prompt.lower()
            if "failure reconciliation" in low or "conserved l2 energy coexist" in low:
                return ExplorerOutput(claims=[ClaimDraft(
                    statement="Conservation of the L2 norm is compatible with unbounded spatial gradients; L2 control of u alone does not control u_x.",
                    claim_type="lemma",
                    derivation="The family u_n(x)=sin(nx) has the same L2 norm on [0,2pi] for every integer n, while |u_n'(0)|=n. Thus bounded amplitude energy and large gradients are not contradictory.",
                    suggested_checks=["energy_gradient_separation"],
                    tags=["burgers", "energy", "gradient", "failure-reuse"],
                )], new_questions=[
                    "Which stronger norm would control the gradient while remaining compatible with Burgers scaling?"
                ])  # type: ignore[return-value]
            if "first zero of the characteristic jacobian" in low or "earliest singular time" in low:
                return ExplorerOutput(claims=[ClaimDraft(
                    statement="If m=min_x u0'(x)<0, the earliest characteristic-Jacobian degeneracy occurs at T=-1/m.",
                    claim_type="lemma",
                    derivation="The Jacobian is J=1+t u0'(xi). For positive t, its earliest zero is obtained at the most negative initial derivative m, hence 1+Tm=0 and T=-1/m.",
                    suggested_checks=["blowup_time_consistency"],
                    tags=["burgers", "characteristics", "singular-time"],
                )])  # type: ignore[return-value]
            if "existence of one negative initial derivative" in low:
                return ExplorerOutput(claims=[ClaimDraft(
                    statement="A Burgers characteristic with initial slope w0<0 has w(t)=w0/(1+t w0) and its gradient diverges at t=-1/w0.",
                    claim_type="lemma",
                    derivation="Solve w'=-w^2 along the characteristic. The denominator reaches zero at positive time precisely when w0<0.",
                    suggested_checks=["riccati_solution"],
                    tags=["burgers", "gradient", "local-blowup"],
                )])  # type: ignore[return-value]
            # Match the more specific gradient branch before the generic word characteristics.
            if "gradient ode" in low or "gradient dynamics" in low:
                return ExplorerOutput(claims=[ClaimDraft(
                    statement="Along a Burgers characteristic, w=u_x obeys w'=-w^2 and hence w(t)=w0/(1+t*w0).",
                    claim_type="lemma",
                    derivation="Differentiate u_t+u u_x=0: w_t+u w_x+w^2=0. Along x'=u this becomes w'=-w^2, whose solution is w0/(1+t w0).",
                    suggested_checks=["riccati_solution"],
                    tags=["burgers", "gradient", "riccati"],
                )])  # type: ignore[return-value]
            if "characteristic map" in low or "characteristics" in low:
                return ExplorerOutput(claims=[ClaimDraft(
                    statement="For inviscid Burgers, characteristics satisfy x(t)=xi+t*u0(xi) and u(x(t),t)=u0(xi).",
                    claim_type="identity",
                    derivation="Along x'(t)=u, d/dt u(x(t),t)=u_t+u u_x=0, so u remains u0(xi); integrating x'=u0(xi) gives x=xi+t u0(xi).",
                    suggested_checks=["burgers_characteristic_jacobian"],
                    tags=["burgers", "characteristics"],
                )])  # type: ignore[return-value]
            if "energy-growth hypothesis" in low or "energy mechanism" in low:
                return ExplorerOutput(claims=[ClaimDraft(
                    statement="Finite-time Burgers gradient blow-up is caused by unbounded growth of the L2 energy of u.",
                    claim_type="conjecture",
                    derivation="Steepening appears to concentrate the solution, suggesting the global L2 norm must diverge before the gradient diverges.",
                    suggested_checks=["burgers_l2_energy"],
                    tags=["burgers", "energy"],
                )])  # type: ignore[return-value]
            return ExplorerOutput(notes="No deterministic mock branch matched.")  # type: ignore[return-value]

        if schema is AssassinOutput:
            low = prompt.lower()
            if "l2 energy" in low and "caused" in low:
                return AssassinOutput(
                    fatal=True,
                    severity="fatal",
                    objections=["For smooth decaying/periodic inviscid Burgers solutions before shock formation, the L2 energy is conserved; gradient blow-up does not require L2 divergence."],
                    reusable_failure="Do not infer derivative singularity from divergence of a conserved global L2 norm.",
                )  # type: ignore[return-value]
            return AssassinOutput(fatal=False, severity="minor", objections=[])  # type: ignore[return-value]

        if schema is VerificationPlan:
            low = prompt.lower()
            checks = []
            if "w'=-w^2" in low or "w0/(1+t*w0)" in low or "w0/(1+t w0)" in low:
                checks.append("riccati_solution")
            if "x(t)=xi+t*u0" in low:
                checks.append("burgers_characteristic_jacobian")
            if "l2 energy" in low and "caused" in low:
                checks.append("burgers_l2_energy")
            if "l2 norm is compatible" in low or "does not control u_x" in low:
                checks.append("energy_gradient_separation")
            if "earliest characteristic-jacobian" in low or "t=-1/m" in low:
                checks.append("blowup_time_consistency")
            return VerificationPlan(checks=checks, rationale="Run deterministic identities relevant to the claim.")  # type: ignore[return-value]

        if schema is SynthesisOutput:
            low = prompt.lower()
            if "w0/(1+t*w0)" in low and "xi+t*u0" in low:
                return SynthesisOutput(
                    claims=[ClaimDraft(
                        statement="If min_x u0'(x)=m<0, the classical inviscid Burgers solution develops gradient blow-up no later than T=-1/m, when the characteristic map loses invertibility.",
                        claim_type="theorem",
                        derivation="The characteristic Jacobian is 1+t*u0'(xi). The gradient formula u_x=u0'(xi)/(1+t*u0'(xi)) diverges when the denominator first vanishes, at T=-1/min u0'.",
                        suggested_checks=["blowup_time_consistency"],
                        tags=["burgers", "blowup", "synthesis"],
                        scope=ClaimScope.ROOT_CANDIDATE,
                    )],
                    cross_branch_links=["Characteristics + gradient dynamics jointly identify the singular time."],
                    notes="Two independently useful branches combine into the benchmark result.",
                )  # type: ignore[return-value]
            return SynthesisOutput(notes="Insufficient verified material to synthesize.")  # type: ignore[return-value]

        if schema is QuestionGeneratorOutput:
            return QuestionGeneratorOutput(
                questions=[
                    ResearchQuestionDraft(
                        question="How can conserved L2 energy coexist with unbounded spatial gradients?",
                        rationale="The failed energy branch exposed a useful distinction between amplitude control and derivative control.",
                        branch_title="Failure reconciliation",
                        expected_information_gain=0.90,
                        expected_impact=0.76,
                        difficulty=0.30,
                        novelty=0.82,
                        spawn_new_branch=True,
                        suggested_strategy="Construct explicit bounded-energy families with increasingly large gradients, then connect the lesson back to the PDE.",
                    ),
                    ResearchQuestionDraft(
                        question="Can the first zero of the characteristic Jacobian determine the earliest singular time exactly?",
                        rationale="This closes the timing link between the characteristic map and the blow-up statement.",
                        branch_title="Characteristics",
                        expected_information_gain=0.78,
                        expected_impact=0.88,
                        difficulty=0.35,
                        novelty=0.35,
                    ),
                    ResearchQuestionDraft(
                        question="Does the Riccati blow-up mechanism require only the existence of one negative initial derivative?",
                        rationale="This tests how local the sufficient blow-up condition is.",
                        branch_title="Gradient dynamics",
                        expected_information_gain=0.72,
                        expected_impact=0.82,
                        difficulty=0.32,
                        novelty=0.45,
                    ),
                ],
                rationale="Resolve the false energy mechanism, then tighten the exact blow-up condition from two independent angles.",
            )  # type: ignore[return-value]

        raise TypeError(f"MockResearchProvider has no fixture for schema {schema.__name__}")
