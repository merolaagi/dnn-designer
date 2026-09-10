from mare.models import EvidenceResult
from mare.verification import run_check


def test_riccati_solution():
    e = run_check("C-test", "riccati_solution")
    assert e.result == EvidenceResult.SUPPORTS


def test_characteristic_jacobian():
    e = run_check("C-test", "burgers_characteristic_jacobian")
    assert e.result == EvidenceResult.SUPPORTS


def test_energy_growth_claim_is_contradicted():
    e = run_check("C-test", "burgers_l2_energy")
    assert e.result == EvidenceResult.CONTRADICTS


def test_blowup_time():
    e = run_check("C-test", "blowup_time_consistency")
    assert e.result == EvidenceResult.SUPPORTS


def test_energy_gradient_separation():
    e = run_check("C-test", "energy_gradient_separation")
    assert e.result == EvidenceResult.SUPPORTS
