import base64
import io
import math

import pytest
from pydantic import ValidationError
from pypdf import PdfWriter

from mare.models import ProblemSpec, ResearchState
from mare_web.studio import (
    DynamicModel,
    Finding,
    PaperAnalysis,
    StudioRequest,
    source_text,
    simulate,
    studio_step,
)


def model(**kwargs):
    return DynamicModel(
        name="Decay",
        variables=["x"],
        derivatives=["-k*x"],
        initial=[1],
        parameters=[{"name": "k", "value": 0.5}],
        **kwargs,
    )


def test_rk4_against_closed_form_and_refinement():
    r = simulate(model(horizon=4))
    assert len(r["points"]) == 201 and r["comparison_complete"]
    assert abs(r["points"][-1]["y"][0] - math.exp(-2)) < 1e-8
    assert r["grid_difference"] < 1e-8
    assert abs(r["first_step"]["k1"][0] + 0.5) < 1e-12


@pytest.mark.parametrize(
    "rhs",
    [
        '__import__("os").system("id")',
        "x.__class__",
        "[x for x in range(9)]",
        "x**x",
        "exp(x, x)",
        "x**99",
        'open("x")',
    ],
)
def test_expression_rejects_code(rhs):
    with pytest.raises((ValueError, ValidationError)):
        DynamicModel(name="bad", variables=["x"], derivatives=[rhs], initial=[1])


def test_failure_is_recorded_not_proved():
    m = DynamicModel(name="blowup", variables=["x"], derivatives=["x**2"], initial=[1], horizon=2)
    r = simulate(m)
    assert r["failure"] and not r["comparison_complete"]
    assert "not a proof" in r["failure"]["meaning"]


def test_coupled_equations():
    m = DynamicModel(
        name="oscillator", variables=["x", "y"], derivatives=["y", "-x"], initial=[1, 0], horizon=3
    )
    final = simulate(m)["points"][-1]["y"]
    assert abs(final[0] - math.cos(3)) < 1e-7
    assert abs(final[1] + math.sin(3)) < 1e-7


def test_invalid_models_and_empty_input():
    with pytest.raises(ValueError):
        DynamicModel(name="bad", variables=["x", "y"], derivatives=["x"], initial=[1])
    with pytest.raises(ValueError):
        DynamicModel(name="bad", variables=["exp"], derivatives=["1"], initial=[1])
    with pytest.raises(ValueError):
        StudioRequest()


def test_text_and_scanned_pdf():
    text = "Assume x > 0. dx/dt = -x."
    assert source_text({"text": text}) == (text, None)
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    stream = io.BytesIO()
    writer.write(stream)
    with pytest.raises(ValueError, match="OCR"):
        source_text({"text": "", "pdf_base64": base64.b64encode(stream.getvalue()).decode()})


async def test_source_anchors_resume_and_proposal():
    state = ResearchState(problem=ProblemSpec(title="t", canonical_statement="test"))
    text = "Assume x > 0.\ndx/dt = -x.\nThis theorem needs a proof."
    state.studio_data = {"input": StudioRequest(text=text, model=model()).model_dump(mode="json")}
    await studio_step(state, None)
    state = ResearchState.model_validate_json(state.model_dump_json())
    await studio_step(state, None)
    for finding in state.studio_data["analysis"]["findings"]:
        assert text[finding["source_start"] : finding["source_end"]] == finding["quote"]
    await studio_step(state, None)
    assert state.current_round == 3 and state.studio_data["experiment"]["comparison_complete"]
    assert not state.claims  # numerical output never becomes a verified theorem


async def test_unsupported_source_does_not_invent_model():
    state = ResearchState(problem=ProblemSpec(title="t", canonical_statement="test"))
    state.studio_data = {
        "input": StudioRequest(text="A difficult open question about prime numbers.").model_dump(mode="json")
    }
    for _ in range(3):
        await studio_step(state, None)
    assert state.studio_data["stage"] == "needs a model"
    assert "experiment" not in state.studio_data


async def test_hallucinated_source_quote_is_flagged():
    class Fake:
        async def generate(self, **kwargs):
            return PaperAnalysis(
                summary="candidate",
                findings=[Finding(kind="rule", quote="not in paper", interpretation="untrusted")],
                model_rationale="none",
                limitations=[],
                next_methods=[],
            )

    state = ResearchState(problem=ProblemSpec(title="t", canonical_statement="test"))
    state.studio_data = {
        "input": StudioRequest(text="A real source sentence.", provider="openai").model_dump(mode="json")
    }
    await studio_step(state, Fake())
    await studio_step(state, Fake())
    assert state.studio_data["analysis"]["findings"][0]["status"] == "unanchored model output"
    assert state.budget.model_calls_used == 1


def test_pdf_text_extraction_has_page_and_hash():
    from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 20 150 Td (Assume a constant rate. dx/dt = -k*x.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    stream = io.BytesIO()
    writer.write(stream)
    text, digest = source_text({"text": "", "pdf_base64": base64.b64encode(stream.getvalue()).decode()})
    assert "[Page 1]" in text and "dx/dt = -k*x" in text
    assert len(digest) == 64
