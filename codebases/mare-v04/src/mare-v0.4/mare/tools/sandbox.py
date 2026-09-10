from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SandboxResult:
    success: bool
    stdout: str
    stderr: str
    returncode: int
    elapsed_ms: float


class DockerPythonSandbox:
    """Runs generated Python in a disposable container with no network.

    Docker remains the security boundary. The host process never eval/execs the
    generated source. The container is read-only, capability-dropped, memory/CPU
    constrained and has a small tmpfs.
    """

    def __init__(self, image: str = "python:3.12-slim", timeout: int = 20):
        self.image = image
        self.timeout = timeout

    @staticmethod
    def available() -> bool:
        return shutil.which("docker") is not None

    def build_command(self, workdir: Path) -> list[str]:
        return [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--memory", "256m",
            "--cpus", "0.5",
            "--pids-limit", "64",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m",
            "-v", f"{workdir}:/work:ro",
            "-w", "/work",
            self.image,
            "python", "-I", "experiment.py",
        ]

    def run(self, code: str) -> SandboxResult:
        if not self.available():
            raise RuntimeError("docker executable not found; secure sandbox unavailable")
        with tempfile.TemporaryDirectory(prefix="mare-sandbox-") as tmp:
            root = Path(tmp)
            (root / "experiment.py").write_text(code, encoding="utf-8")
            cmd = self.build_command(root)
            start = time.perf_counter()
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout, check=False)
            elapsed = (time.perf_counter() - start) * 1000
            return SandboxResult(proc.returncode == 0, proc.stdout, proc.stderr, proc.returncode, elapsed)
