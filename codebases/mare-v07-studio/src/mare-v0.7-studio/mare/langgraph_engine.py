"""Optional LangGraph mapping of the MARE v0.3 state machine.

The native ResearchEngine remains the reference implementation. This adapter
maps the same phases to explicit StateGraph nodes/edges for durable orchestration.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .engine import ResearchEngine
from .models import ResearchState


class GraphState(TypedDict):
    research: ResearchState
    rounds_left: int
    round_start_claim_count: int


def build_graph(engine: ResearchEngine):
    async def initialize(s: GraphState):
        state = s["research"]
        if not state.branches:
            await engine.initialize(state)
        state.current_round += 1
        return {"research": state, "round_start_claim_count": len(state.claims)}

    async def explore(s: GraphState):
        state = s["research"]
        await engine.exploration_round(state)
        return {"research": state}

    async def review_verify(s: GraphState):
        state = s["research"]
        new_claims = state.claims[s["round_start_claim_count"]:]
        if new_claims:
            await engine.adversarial_review(state, new_claims)
            await engine.verify(state, new_claims)
            await engine.enrich_literature(state, new_claims)
        return {"research": state}

    async def synthesize(s: GraphState):
        state = s["research"]
        claims = await engine.synthesize(state)
        if claims:
            await engine.adversarial_review(state, claims)
            await engine.verify(state, claims)
            await engine.enrich_literature(state, claims)
        engine.navigator.score(state)
        engine.navigator.update_lifecycle(state)
        return {"research": state}

    async def questions(s: GraphState):
        state = s["research"]
        await engine.generate_questions(state)
        return {"research": state}

    def advance(s: GraphState):
        state = s["research"]
        left = s["rounds_left"] - 1
        if left > 0:
            engine.prepare_next_round(state)
        return {"research": state, "rounds_left": left}

    def route(s: GraphState):
        return "again" if s["rounds_left"] > 0 else "done"

    g = StateGraph(GraphState)
    g.add_node("initialize_round", initialize)
    g.add_node("explore", explore)
    g.add_node("review_verify", review_verify)
    g.add_node("synthesize", synthesize)
    g.add_node("questions", questions)
    g.add_node("advance", advance)
    g.add_edge(START, "initialize_round")
    g.add_edge("initialize_round", "explore")
    g.add_edge("explore", "review_verify")
    g.add_edge("review_verify", "synthesize")
    g.add_edge("synthesize", "questions")
    g.add_edge("questions", "advance")
    g.add_conditional_edges("advance", route, {"again": "initialize_round", "done": END})
    return g.compile()
