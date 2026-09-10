from __future__ import annotations

import sympy as sp

from .models import Evidence, EvidenceResult


def _evidence(claim_id: str, check: str, ok: bool, details: str) -> Evidence:
    return Evidence(
        claim_id=claim_id,
        evidence_type="symbolic",
        result=EvidenceResult.SUPPORTS if ok else EvidenceResult.CONTRADICTS,
        verifier=f"sympy:{check}",
        reproducible=True,
        details=details,
    )


def run_check(claim_id: str, check: str) -> Evidence:
    t, w0 = sp.symbols("t w0")

    if check == "riccati_solution":
        w = w0 / (1 + t * w0)
        residual = sp.simplify(sp.diff(w, t) + w**2)
        return _evidence(claim_id, check, residual == 0, f"Residual d/dt(w)+w^2 simplifies to {residual}.")

    if check == "burgers_characteristic_jacobian":
        xi = sp.symbols("xi")
        u0 = sp.Function("u0")
        x = xi + t * u0(xi)
        jac = sp.diff(x, xi)
        expected = 1 + t * sp.diff(u0(xi), xi)
        residual = sp.simplify(jac - expected)
        return _evidence(claim_id, check, residual == 0, f"Characteristic Jacobian check residual is {residual}.")

    if check == "burgers_l2_energy":
        # For smooth periodic/decaying solutions: d/dt ∫u²/2 dx = -∫(u³/3)_x dx = 0.
        # We encode the integration-by-parts identity symbolically as the derivative of u^3/3.
        u = sp.symbols("u")
        flux = u**3 / 3
        flux_derivative = sp.diff(flux, u)
        ok = sp.simplify(flux_derivative - u**2) == 0
        # The claim being tested says L2 growth causes blow-up; conservation contradicts it.
        return Evidence(
            claim_id=claim_id,
            evidence_type="symbolic",
            result=EvidenceResult.CONTRADICTS if ok else EvidenceResult.INCONCLUSIVE,
            verifier=f"sympy:{check}",
            reproducible=True,
            details="For smooth solutions before singularity, u^2 u_x=(u^3/3)_x, so the spatial integral is a boundary term and L2 energy is conserved under periodic/decaying conditions.",
        )


    if check == "energy_gradient_separation":
        x, n = sp.symbols("x n", real=True, positive=True, integer=True)
        u = sp.sin(n * x)
        l2 = sp.integrate(u**2, (x, 0, 2 * sp.pi))
        du = sp.diff(u, x)
        # L2(u) is independent of n while max |u_x| scales like n.
        ok = sp.simplify(l2 - sp.pi) == 0 and sp.simplify(du.subs(x, 0) - n) == 0
        return _evidence(
            claim_id,
            check,
            ok,
            f"For u_n=sin(nx), integral_0^(2pi) u_n^2 dx = {l2}, while u_n'(0) = {sp.simplify(du.subs(x, 0))}.",
        )

    if check == "blowup_time_consistency":
        m = sp.symbols("m", negative=True, nonzero=True)
        T = -1 / m
        denominator = sp.simplify(1 + T * m)
        return _evidence(claim_id, check, denominator == 0, f"At T=-1/m, 1+T*m simplifies to {denominator}.")

    return Evidence(
        claim_id=claim_id,
        evidence_type="symbolic",
        result=EvidenceResult.INCONCLUSIVE,
        verifier=f"unknown:{check}",
        reproducible=False,
        details=f"Unknown verification check: {check}",
    )
