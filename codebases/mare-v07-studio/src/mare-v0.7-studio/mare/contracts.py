from pydantic import BaseModel, Field

from .models import ClaimDraft


class BranchPlan(BaseModel):
    title: str
    research_question: str
    strategy: str
    novelty: float = 0.5


class TaskPlan(BaseModel):
    branch_title: str
    title: str
    objective: str


class DirectorOutput(BaseModel):
    branches: list[BranchPlan] = Field(default_factory=list)
    tasks: list[TaskPlan] = Field(default_factory=list)
    rationale: str = ""


class ExplorerOutput(BaseModel):
    claims: list[ClaimDraft] = Field(default_factory=list)
    new_questions: list[str] = Field(default_factory=list)
    notes: str = ""


class AssassinOutput(BaseModel):
    fatal: bool = False
    severity: str = "minor"
    objections: list[str] = Field(default_factory=list)
    reusable_failure: str | None = None


class VerificationPlan(BaseModel):
    checks: list[str] = Field(default_factory=list)
    rationale: str = ""


class SynthesisOutput(BaseModel):
    claims: list[ClaimDraft] = Field(default_factory=list)
    cross_branch_links: list[str] = Field(default_factory=list)
    notes: str = ""


class ResearchQuestionDraft(BaseModel):
    question: str
    rationale: str = ""
    branch_title: str | None = None
    expected_information_gain: float = 0.5
    expected_impact: float = 0.5
    difficulty: float = 0.5
    novelty: float = 0.5
    spawn_new_branch: bool = False
    suggested_strategy: str | None = None


class QuestionGeneratorOutput(BaseModel):
    questions: list[ResearchQuestionDraft] = Field(default_factory=list)
    rationale: str = ""
