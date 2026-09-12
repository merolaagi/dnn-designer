"""What each problem actually says, what it needs, and what has been tried.

Written for somebody who can read mathematics. There is no encouragement here
and no scaffolding: the statement with its quantifiers, the results it sits on,
the attempts that failed and the specific reason each one failed. That last
part is the valuable one and the hardest to find — the literature records what
worked, and the folklore records what did not.

Two rules held throughout:

    a barrier is named with its theorem      "relativization, Baker-Gill-
                                             Solovay 1975", not "it is hard"
    solved problems say how they were solved Poincare is not open, and the
                                             method that closed it is the
                                             interesting part

Where machine learning has genuinely contributed to pure mathematics it is
said, with the case; where it has not, that is said too. Both are short lists
and pretending otherwise would waste the reader's time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

DOSSIERS: List[Dict[str, Any]] = [
    {
        "id": "poincare",
        "name": "The Poincaré conjecture",
        "status": "solved",
        "one_line": "Every simply connected closed 3-manifold is homeomorphic "
                    "to S³.",
        "statement": "Let M be a compact, connected 3-manifold without "
                     "boundary. If every loop in M contracts to a point "
                     "(π₁(M) = 1), then M is homeomorphic to the 3-sphere. "
                     "Note what is not assumed: no smoothness of a map, no "
                     "embedding, no metric. In dimension 3 smooth, PL and "
                     "topological categories coincide (Moise), so the smooth "
                     "statement is equivalent.",
        "why_hard": "The obvious approach — build the homeomorphism by hand — "
                    "fails because a simply connected 3-manifold can be "
                    "presented in wildly different ways, and no algorithmic "
                    "normal form was known. Dimension 3 is the worst case: too "
                    "rigid for the surgery-theoretic h-cobordism argument that "
                    "settles n ≥ 5, and too low for the Whitney trick, which "
                    "needs 2 + 2 < n.",
        "lineage": [
            ("Fundamental group", "π₁ is the invariant the hypothesis is "
             "stated in. Simply connected means π₁ trivial."),
            ("Smale, n ≥ 5 (1961)", "Proved by handle cancellation: the "
             "h-cobordism theorem. The Whitney trick needs enough room, which "
             "high dimensions have."),
            ("Freedman, n = 4 (1982)", "Topological category only, via Casson "
             "handles. The smooth 4-dimensional Poincaré conjecture is still "
             "open — the one case nobody has."),
            ("Thurston geometrization", "Every closed 3-manifold decomposes "
             "into pieces carrying one of eight geometries. Poincaré is the "
             "special case where the piece is spherical."),
            ("Hamilton, Ricci flow (1982)", "∂g/∂t = −2 Ric(g). A heat "
             "equation for the metric: curvature spreads out. Hamilton proved "
             "it works when Ric > 0, and identified the obstacle — "
             "singularities in finite time."),
            ("Perelman (2002–03)", "Three preprints. Ricci flow with surgery, "
             "the W-entropy and the no-local-collapsing theorem, "
             "classification of κ-solutions, and finite extinction time for "
             "simply connected manifolds."),
        ],
        "attempts": [
            ("Whitehead, 1934", "Announced a proof, then found his own "
             "counterexample to a step: the Whitehead manifold, a contractible "
             "open 3-manifold not homeomorphic to ℝ³. Contractible does not "
             "imply standard, which is the whole difficulty in miniature."),
            ("Surgery theory", "Works from n ≥ 5 and does not descend. The "
             "Whitney trick needs 2 + 2 < n to separate intersections; in "
             "dimension 3 there is no room."),
            ("Combinatorial / algorithmic attacks", "Recognising S³ from a "
             "triangulation is decidable (Rubinstein, Thompson) but that gives "
             "no classification, and the certificate does not prove the "
             "conjecture."),
            ("Ricci flow without surgery", "Fails: singularities form. "
             "Hamilton's programme was blocked for two decades on exactly "
             "this, and Perelman's contribution was the analysis that tames "
             "them — κ-noncollapsing rules out the cigar soliton, so the "
             "blow-up limits are classifiable."),
        ],
        "settled_by": "Perelman, 2002–03; verified in detail by Kleiner–Lott, "
                      "Morgan–Tian and Cao–Zhu over the following four years.",
        "ml_note": "No part of this is a pattern-recognition problem. The "
                   "difficulty was analytic control of a geometric flow near "
                   "singularities, and nothing in current machine learning "
                   "addresses that. Claiming otherwise would be selling "
                   "something.",
        "open_relative": "The smooth 4-dimensional Poincaré conjecture remains "
                         "open, and is a genuinely live target.",
    },
    {
        "id": "riemann",
        "name": "The Riemann hypothesis",
        "status": "open",
        "one_line": "Every non-trivial zero of ζ(s) has real part ½.",
        "statement": "ζ(s) = Σ n^(−s) for Re(s) > 1, continued meromorphically "
                     "to ℂ with a simple pole at s = 1. It has trivial zeros "
                     "at s = −2, −4, …. The hypothesis: every other zero lies "
                     "on Re(s) = ½. Equivalent: π(x) = li(x) + O(√x log x), "
                     "which is the sharpest possible error term in the prime "
                     "number theorem.",
        "why_hard": "There is no known construction that forces zeros onto a "
                    "line. The functional equation gives a symmetry about "
                    "Re(s) = ½ but symmetry does not imply concentration. "
                    "Every unconditional result bounds the zero-free region or "
                    "counts a proportion on the line; none can reach all of "
                    "them, because the methods are ultimately about averages.",
        "lineage": [
            ("Euler product", "ζ(s) = Π (1 − p^(−s))^(−1). The bridge between "
             "an analytic object and the primes; everything downstream depends "
             "on it."),
            ("Functional equation", "ξ(s) = ξ(1 − s) after completing with Γ "
             "and π factors. The critical line is the axis of this symmetry."),
            ("Prime number theorem (1896)", "Hadamard and de la Vallée "
             "Poussin, from ζ having no zeros on Re(s) = 1. The hypothesis is "
             "the quantitative strengthening."),
            ("Explicit formula", "von Mangoldt: ψ(x) = x − Σ_ρ x^ρ/ρ − …. Each "
             "zero contributes an oscillation of size x^Re(ρ). This is why the "
             "real part is the whole question."),
            ("Weil conjectures", "RH proved for zeta functions of varieties "
             "over finite fields — Weil for curves, Deligne in general. The "
             "geometry supplies a cohomology and a positivity that the "
             "classical case lacks."),
        ],
        "attempts": [
            ("Hilbert–Pólya", "If the zeros were eigenvalues of a self-adjoint "
             "operator, reality of the spectrum would give RH. Montgomery's "
             "pair correlation and Odlyzko's computations match random matrix "
             "statistics (GUE), which is striking evidence — but nobody has "
             "produced the operator."),
            ("de Branges", "Several announced proofs via Hilbert spaces of "
             "entire functions; Conrey and Li produced explicit obstructions "
             "to the approach as stated."),
            ("Zero-density and proportion results", "Selberg, Levinson, "
             "Conrey: a positive proportion of zeros lie on the line — more "
             "than 40% is known. These methods are averaging arguments and do "
             "not reach 100%."),
            ("Connes' trace formula", "Reformulates RH as a positivity "
             "statement on an adelic space. The reformulation is real; the "
             "positivity is as hard as the original."),
            ("Numerical verification", "Over 10¹³ zeros checked on the line. "
             "This is not evidence of the kind that matters: Skewes-type "
             "phenomena show number-theoretic patterns can first fail beyond "
             "any feasible computation."),
        ],
        "ml_note": "Machine learning has contributed to pure mathematics by "
                   "generating conjectures from data — Davies et al. (Nature, "
                   "2021) found a genuine relation between knot invariants "
                   "this way, later proved by hand. That is the realistic "
                   "shape of a contribution here: finding structure in "
                   "computed zeros or L-function data that a human then "
                   "proves. It is not a route to the theorem.",
    },
    {
        "id": "navier_stokes",
        "name": "Navier–Stokes existence and smoothness",
        "status": "open",
        "one_line": "Do smooth solutions of the 3D incompressible "
                    "Navier–Stokes equations exist for all time?",
        "statement": "Given smooth, divergence-free initial data u₀ on ℝ³ (or "
                     "the torus) with finite energy, does the system "
                     "∂_t u + (u·∇)u = −∇p + νΔu, ∇·u = 0 admit a smooth "
                     "solution for all t > 0? Or does some initial datum lead "
                     "to blow-up in finite time? Either answer settles it.",
        "why_hard": "The equation is supercritical in three dimensions. The "
                    "scaling u_λ(x,t) = λu(λx, λ²t) leaves the equation "
                    "invariant but the energy norm is not scale-invariant — it "
                    "grows as λ^(−1/2) — so the conserved quantity is weaker "
                    "than the scale at which trouble occurs. Every known "
                    "global result is either in 2D, for small data, or under "
                    "an assumed criterion.",
        "lineage": [
            ("Leray (1934)", "Weak solutions exist globally, with finite "
             "energy. Uniqueness and regularity of these are unknown — the gap "
             "has been open for ninety years."),
            ("Energy inequality", "‖u(t)‖² + 2ν∫‖∇u‖² ≤ ‖u₀‖². The only "
             "unconditional global control there is, and it is supercritical."),
            ("Ladyzhenskaya–Prodi–Serrin criteria", "Regularity follows if u ∈ "
             "L^p_t L^q_x with 2/p + 3/q ≤ 1. Conditional, but it says exactly "
             "which norm would have to blow up."),
            ("Caffarelli–Kohn–Nirenberg (1982)", "Partial regularity: the "
             "singular set of a suitable weak solution has parabolic "
             "1-dimensional Hausdorff measure zero. Singularities, if they "
             "exist, are very small."),
            ("Beale–Kato–Majda", "Blow-up at T requires ∫₀^T ‖ω(t)‖_∞ dt = ∞. "
             "Turns the question into control of vorticity."),
        ],
        "attempts": [
            ("Energy methods alone", "Cannot work in 3D by scaling: the energy "
             "norm is supercritical for the nonlinearity. This is not a "
             "technical gap, it is the reason the problem is open."),
            ("Tao (2016)", "Constructed an averaged Navier–Stokes equation, "
             "obeying the same energy identity and scaling, whose solutions "
             "blow up in finite time. This is a barrier result: any proof of "
             "global regularity must use something beyond energy and scaling, "
             "because those alone are consistent with blow-up."),
            ("Numerical searches for blow-up", "Repeated candidate scenarios "
             "— vortex reconnection, axisymmetric with swirl — have been "
             "proposed and then resolved as near-singular but smooth. "
             "Hou–Luo's 2014 boundary scenario remains the most persuasive "
             "numerical candidate."),
            ("Convex integration", "Buckmaster–Vicol proved non-uniqueness of "
             "weak solutions with finite energy. It does not touch the smooth "
             "problem but it shows the weak class is far too large to be a "
             "route."),
        ],
        "ml_note": "The plausible contribution is finding a blow-up scenario "
                   "numerically — this is a search over initial data with a "
                   "computable objective, which is a shape machine learning "
                   "suits. Neural-network-assisted searches for self-similar "
                   "blow-up in related equations have produced candidate "
                   "profiles. What no such search can do is prove regularity; "
                   "it can only find the counterexample, if there is one.",
    },
    {
        "id": "p_np",
        "name": "P versus NP",
        "status": "open",
        "one_line": "Can every problem whose solution is quickly checkable be "
                    "quickly solved?",
        "statement": "Is P = NP? Equivalently: does SAT have a "
                     "polynomial-time algorithm? The interesting direction is "
                     "P ≠ NP, which requires a superpolynomial lower bound on "
                     "circuit or time complexity for an explicit problem.",
        "why_hard": "Three barrier theorems, each ruling out a whole style of "
                    "argument. Any proof must avoid all three, and no current "
                    "technique does.",
        "lineage": [
            ("Cook–Levin (1971)", "SAT is NP-complete: one problem stands for "
             "all of them."),
            ("Karp (1972)", "Twenty-one problems reduced, establishing that "
             "NP-completeness is ubiquitous rather than exotic."),
            ("Circuit complexity", "Reformulates the question as a lower bound "
             "on circuit size, which is concrete and still out of reach for "
             "any explicit function."),
        ],
        "attempts": [
            ("Diagonalisation", "Blocked by relativization: Baker, Gill and "
             "Solovay (1975) gave oracles A, B with P^A = NP^A and P^B ≠ NP^B. "
             "Any argument that relativizes cannot settle it."),
            ("Combinatorial circuit lower bounds", "Blocked by natural proofs: "
             "Razborov and Rudich (1994) showed that a lower-bound argument "
             "which is constructive and large would break pseudorandom "
             "generators, contradicting standard cryptographic assumptions."),
            ("Arithmetisation", "Blocked by algebrization: Aaronson and "
             "Wigderson (2008) extended the relativization barrier to cover "
             "the algebraic techniques that beat it, such as IP = PSPACE."),
            ("Geometric complexity theory", "Mulmuley and Sohoni's programme "
             "— representation theory of orbit closures — is designed to avoid "
             "all three barriers. Ikenmeyer and Panova later showed the "
             "specific multiplicity-based obstructions it hoped for do not "
             "exist, so the programme is alive but not in its original form."),
        ],
        "ml_note": "Nothing. Learning a separation is not a coherent goal, and "
                   "the barriers are statements about proof techniques rather "
                   "than about search difficulty.",
    },
    {
        "id": "collatz",
        "name": "The Collatz conjecture",
        "status": "open",
        "one_line": "Iterating n → n/2 or 3n+1 reaches 1 from every positive "
                    "integer.",
        "statement": "Define T(n) = n/2 for even n, 3n+1 for odd n. Does the "
                     "orbit of every n ≥ 1 reach 1? Two things must be ruled "
                     "out: divergence to infinity, and a cycle other than "
                     "1 → 4 → 2 → 1.",
        "why_hard": "The map mixes two incompatible structures — halving is "
                    "2-adic, tripling is not — and no invariant is known that "
                    "decreases along orbits. Conway showed a natural "
                    "generalisation (FRACTRAN-style maps) is undecidable, "
                    "which suggests the difficulty is not a missing trick.",
        "lineage": [
            ("Stopping time", "Almost all n have a stopping time: they drop "
             "below their starting value. Terras, Everett, 1976–77."),
            ("Cycle bounds", "Any non-trivial cycle must be extremely long — "
             "continued-fraction bounds on log 3 / log 2 force it, and "
             "computation pushes the bound further."),
            ("Tao (2019)", "Almost all orbits attain almost bounded values: "
             "for any f → ∞, the orbit of almost every n drops below f(n). "
             "The strongest general result, and it is still a statement about "
             "almost all n."),
        ],
        "attempts": [
            ("Exhaustive computation", "Verified beyond 2^68. This settles "
             "nothing, since a divergent orbit or a long cycle could begin "
             "anywhere above."),
            ("Invariant hunting", "No monotone quantity is known. Attempts "
             "founder because the 3n+1 step can increase any natural measure "
             "of size."),
            ("Undecidability by analogy", "Conway's result on generalised "
             "Collatz maps means no general algorithm decides such problems, "
             "so any proof must use the specific arithmetic of 3 and 2."),
        ],
        "ml_note": "This is the rare case where search is genuinely useful. "
                   "Pattern-finding over orbit data is cheap, and machine "
                   "learning has been used to predict stopping times. No "
                   "result of consequence has come out of it, and a "
                   "conjecture found this way would still need proving — but "
                   "the experiment is not absurd, unlike for Poincaré.",
    },
    {
        "id": "fermat",
        "name": "Fermat's last theorem",
        "status": "solved",
        "one_line": "No positive integers satisfy aⁿ + bⁿ = cⁿ for n > 2.",
        "statement": "For n ≥ 3 there are no positive integers a, b, c with "
                     "aⁿ + bⁿ = cⁿ. It suffices to prove it for n = 4 and for "
                     "odd primes n = p.",
        "why_hard": "Three centuries of attacks on the equation itself "
                    "produced results for individual exponents and no general "
                    "method. What eventually worked did not look at the "
                    "equation at all.",
        "lineage": [
            ("Kummer, regular primes", "Ideal class groups and cyclotomic "
             "fields; the failure of unique factorisation is precisely what "
             "breaks the naive descent."),
            ("Frey curve (1984)", "A solution aᵖ + bᵖ = cᵖ gives the elliptic "
             "curve y² = x(x − aᵖ)(x + bᵖ), whose invariants are too good to "
             "be true."),
            ("Ribet's theorem (1986)", "The Frey curve cannot be modular. So "
             "Taniyama–Shimura implies Fermat — the reduction that made the "
             "problem attackable."),
            ("Wiles, Taylor–Wiles (1994–95)", "Modularity of semistable "
             "elliptic curves over ℚ, via a numerical criterion for "
             "isomorphism of deformation rings and Hecke algebras."),
        ],
        "attempts": [
            ("Descent on the equation", "Works for n = 3, 4, 5, 7 with "
             "increasing effort and does not generalise."),
            ("Kummer's approach", "Settles regular primes; irregular primes "
             "remain, and infinitely many are expected."),
            ("The first Wiles manuscript", "Contained a gap in the Euler "
             "system argument, found in review, closed a year later with "
             "Taylor by a different route. The gap is worth knowing about: "
             "the first version of a hard proof is usually wrong somewhere."),
        ],
        "settled_by": "Wiles (1995), with Taylor, via modularity.",
        "ml_note": "The proof is a structural reduction, not a search. Nothing "
                   "in it would have been found by pattern-matching over "
                   "solutions, because there are no solutions to pattern-match.",
    },
]


def listing() -> List[Dict[str, Any]]:
    return [{"id": d["id"], "name": d["name"], "status": d["status"],
             "one_line": d["one_line"]} for d in DOSSIERS]


def get(problem_id: str) -> Optional[Dict[str, Any]]:
    for entry in DOSSIERS:
        if entry["id"] == problem_id:
            return entry
    return None


def suggest(title: str, statement: str) -> Optional[Dict[str, Any]]:
    """Recognise a stated problem as one of these, by its own vocabulary."""
    text = f"{title} {statement}".lower()
    keys = {
        "poincare": ["poincar", "simply connected", "3-manifold",
                     "three-manifold", "ricci flow"],
        "riemann": ["riemann", "zeta", "critical line", "nontrivial zero",
                    "non-trivial zero"],
        "navier_stokes": ["navier", "stokes", "incompressible", "blow-up",
                          "blow up", "vorticity"],
        "p_np": ["p vs np", "p versus np", "np-complete", "sat", "polynomial "
                 "time"],
        "collatz": ["collatz", "3n+1", "3n + 1", "hailstone", "syracuse"],
        "fermat": ["fermat", "a^n + b^n", "last theorem"],
    }
    best, score = None, 0
    for entry_id, words in keys.items():
        hits = sum(1 for w in words if w in text)
        if hits > score:
            best, score = entry_id, hits
    return get(best) if best and score else None
