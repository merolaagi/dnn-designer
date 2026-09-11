from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator

from mare.models import ResearchState, DependencyEdge

RunStatus = Literal["queued", "running", "cancelling", "cancelled", "completed", "failed", "budget_exhausted"]


class CreateRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(default="Burgers gradient blow-up", min_length=1, max_length=200)
    provider: Literal["mock", "openai"] = "mock"
    statement: str | None = Field(default=None, min_length=10, max_length=12000)
    assumptions: list[str] = Field(default_factory=list, max_length=30)
    success_conditions: list[str] = Field(default_factory=list, max_length=30)
    rounds: int = Field(default=2, ge=1, le=30)
    policy: Literal["hybrid", "heuristic"] = "hybrid"
    max_model_calls: int = Field(default=250, ge=1, le=2000)
    max_reserved_output_tokens: int = Field(default=1000000, ge=1, le=8000000)
    max_tool_calls: int = Field(default=2000, ge=1, le=10000)

    @model_validator(mode="after")
    def provider_problem(self):
        if not self.title.strip():
            raise ValueError("Title cannot be blank")
        if self.provider == "mock" and (self.statement or self.assumptions or self.success_conditions):
            raise ValueError("Mock provider only supports the immutable Burgers benchmark")
        if self.provider == "openai" and not self.statement:
            raise ValueError("OpenAI runs require a research statement")
        return self


class UTCOutput(BaseModel):
    @field_validator("created_at", "updated_at", check_fields=False)
    @classmethod
    def utc_dates(cls, value):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


class RunSummary(UTCOutput):
    workflow: Literal["research", "studio"] = "research"
    model_config = ConfigDict(from_attributes=True)
    id: str
    title: str
    status: RunStatus
    provider: str
    target_rounds: int
    current_round: int
    attempts: int
    created_at: datetime
    updated_at: datetime
    error: str | None


class RunDetail(RunSummary):
    snapshot: ResearchState


class EventOut(UTCOutput):
    model_config = ConfigDict(from_attributes=True)
    id: int
    run_id: str
    kind: str
    payload: dict
    created_at: datetime


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorDetail


class ServiceSettings(BaseModel):
    version: str
    engine_version: str
    authentication: str
    providers: list[str]
    openai_model: str | None
    max_attempts: int
    round_timeout_seconds: int
    checkpoint_boundary: str
    workspace_mode: str


class GraphNode(BaseModel):
    id: str
    statement: str
    status: str
    completeness: float


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[DependencyEdge]
    has_cycle: bool
    candidate_roots: list[str]


class HealthResponse(BaseModel):
    status: str
