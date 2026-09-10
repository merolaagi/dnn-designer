from __future__ import annotations

import re

from .models import AssumptionAudit, AuditStatus, Claim, ClaimScope, ProblemSpec

_TOKEN = re.compile(r"[a-z0-9_]+")


def _norm(s: str) -> str:
    return " ".join(_TOKEN.findall(s.lower()))


def _loosely_matches(extra: str, allowed: str) -> bool:
    a, b = _norm(extra), _norm(allowed)
    if not a or not b:
        return False
    return a in b or b in a


class AssumptionFirewall:
    """Prevents a root claim from silently solving a narrower problem.

    Local lemmas may use auxiliary assumptions; root candidates may not add an
    extra domain/symmetry/regularity hypothesis unless it is already present in
    the immutable ProblemSpec. This is intentionally conservative.
    """

    def audit(self, problem: ProblemSpec, claim: Claim) -> AssumptionAudit:
        allowed = [*problem.assumptions, *problem.scope_constraints]
        matched: list[str] = []
        violations: list[str] = []

        for extra in claim.extra_assumptions:
            match = next((a for a in allowed if _loosely_matches(extra, a)), None)
            if match:
                matched.append(match)
            else:
                violations.append(f"extra assumption not present in ProblemSpec: {extra}")

        for shortcut in problem.disallowed_shortcuts:
            n = _norm(shortcut)
            if n and n in _norm(claim.derivation):
                violations.append(f"derivation appears to invoke disallowed shortcut: {shortcut}")

        if claim.scope == ClaimScope.ROOT_CANDIDATE and violations:
            status = AuditStatus.BLOCK
        elif violations:
            status = AuditStatus.WARN
        else:
            status = AuditStatus.PASS

        return AssumptionAudit(
            claim_id=claim.id,
            status=status,
            extra_assumptions=claim.extra_assumptions,
            matched_problem_constraints=matched,
            violations=violations,
            notes=(
                "Root candidate matches the immutable problem scope."
                if status == AuditStatus.PASS and claim.scope == ClaimScope.ROOT_CANDIDATE
                else "Auxiliary assumptions reviewed against immutable problem scope."
            ),
        )
