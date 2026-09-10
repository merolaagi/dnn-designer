from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Protocol

from .models import Claim, FormalizationRecord, FormalizationStatus


class LeanService(Protocol):
    name: str

    def check(self, claim: Claim, source_code: str) -> FormalizationRecord:
        ...


class MockLeanService:
    name = "mock-lean"

    def __init__(self, should_pass: bool = True):
        self.should_pass = should_pass

    def check(self, claim: Claim, source_code: str) -> FormalizationRecord:
        return FormalizationRecord(
            claim_id=claim.id,
            status=FormalizationStatus.PASSED if self.should_pass else FormalizationStatus.FAILED,
            prover=self.name,
            source_code=source_code,
            stdout="mock kernel accepted" if self.should_pass else "",
            stderr="" if self.should_pass else "mock kernel rejected",
        )


class SubprocessLeanService:
    """Checks Lean source using `lake env lean` inside an existing Lean project."""

    name = "lean4"

    def __init__(self, project_dir: str | Path, timeout: int = 30):
        self.project_dir = Path(project_dir)
        self.timeout = timeout

    def available(self) -> bool:
        return shutil.which("lake") is not None and self.project_dir.exists()

    def check(self, claim: Claim, source_code: str) -> FormalizationRecord:
        if not self.available():
            return FormalizationRecord(
                claim_id=claim.id,
                status=FormalizationStatus.SKIPPED,
                prover=self.name,
                source_code=source_code,
                stderr="Lean/Lake is not available or the configured project directory does not exist.",
            )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".lean", prefix="Mare_", dir=self.project_dir, delete=False, encoding="utf-8"
        ) as f:
            f.write(source_code)
            path = Path(f.name)
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                ["lake", "env", "lean", path.name],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
            elapsed = (time.perf_counter() - start) * 1000
            return FormalizationRecord(
                claim_id=claim.id,
                status=FormalizationStatus.PASSED if proc.returncode == 0 else FormalizationStatus.FAILED,
                prover=self.name,
                source_code=source_code,
                stdout=proc.stdout,
                stderr=proc.stderr,
                elapsed_ms=elapsed,
                artifact_path=str(path),
            )
        finally:
            # Proof source is persisted in the FormalizationRecord; avoid leaving
            # uncontrolled temporary source files in the Lean workspace.
            path.unlink(missing_ok=True)
