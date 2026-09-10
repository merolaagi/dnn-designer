from .models import ProblemSpec


def burgers_blowup_problem() -> ProblemSpec:
    return ProblemSpec(
        title="Inviscid Burgers finite-time gradient blow-up",
        canonical_statement=(
            "Consider the one-dimensional inviscid Burgers equation u_t + u u_x = 0 "
            "with smooth initial data u(x,0)=u0(x). Determine a rigorous mechanism and "
            "condition under which the classical solution develops finite-time gradient blow-up."
        ),
        definitions=[
            "A classical solution is C^1 in the spacetime region under discussion.",
            "Gradient blow-up means |u_x| becomes unbounded while u can remain bounded.",
        ],
        assumptions=[
            "u0 is smooth.",
            "Use either periodic data or sufficient decay when invoking spatial integral identities.",
        ],
        success_conditions=[
            "Identify a condition on u0' implying finite-time gradient blow-up.",
            "Derive the singular time from the solution mechanism.",
            "Do not confuse bounded amplitude with bounded gradient.",
        ],
        disallowed_shortcuts=[
            "Do not cite the standard closed-form Burgers theorem as a black box.",
            "Do not claim that L2 energy divergence is necessary for gradient blow-up.",
        ],
    )
