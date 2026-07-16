from __future__ import annotations

import dataclasses
import subprocess


@dataclasses.dataclass(frozen=True)
class IsaacEulaProbe:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    manual_action_required: bool


def probe_isaac_simulation_app(python_executable: str = ".venv_issac/bin/python", timeout_seconds: int = 30) -> IsaacEulaProbe:
    code = (
        "from isaacsim import SimulationApp\n"
        "print('ABOUT_TO_START_SIMULATION_APP')\n"
        "app = SimulationApp({'headless': True, 'create_new_stage': True})\n"
        "print('SIMULATION_APP_STARTED')\n"
        "app.close()\n"
        "print('SIMULATION_APP_CLOSED')\n"
    )
    command = [python_executable, "-c", code]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout_seconds, check=False)
    combined = result.stdout + "\n" + result.stderr
    manual = "Do you accept the EULA" in combined or "NVIDIA OMNIVERSE LICENSE AGREEMENT" in combined
    return IsaacEulaProbe(command, result.returncode, result.stdout, result.stderr, manual)

