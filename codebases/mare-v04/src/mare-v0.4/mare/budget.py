from __future__ import annotations

from .models import ResearchState, ResearchTask


class BudgetExhausted(RuntimeError):
    pass


class BudgetController:
    """Provider-neutral hard budget enforcement.

    Model reservations use the requested max output tokens. This deliberately
    over-approximates usage when providers do not expose actual accounting.
    """

    def check_round(self, state: ResearchState, round_no: int) -> None:
        if round_no > state.budget.max_rounds:
            raise BudgetExhausted(f"round {round_no} exceeds max_rounds={state.budget.max_rounds}")

    def can_schedule(self, state: ResearchState, task: ResearchTask) -> bool:
        return state.budget.can_reserve_model_call(task.token_budget)

    def reserve_model_call(self, state: ResearchState, max_tokens: int) -> None:
        try:
            state.budget.reserve_model_call(max_tokens)
        except RuntimeError as exc:
            raise BudgetExhausted(str(exc)) from exc

    def reserve_tool_call(self, state: ResearchState, count: int = 1) -> None:
        try:
            state.budget.reserve_tool(count)
        except RuntimeError as exc:
            raise BudgetExhausted(str(exc)) from exc
