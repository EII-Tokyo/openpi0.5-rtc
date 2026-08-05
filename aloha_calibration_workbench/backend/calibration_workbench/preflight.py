from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel
from pydantic import Field

from .models import CandidateRegistry
from .models import IssueSeverity
from .models import OwnershipState
from .models import PreflightCamera
from .models import PreflightIssue
from .models import PreflightReport
from .models import PreflightStatus


class CameraObservation(BaseModel):
    serial: str
    product_name: str
    firmware: str
    recommended_firmware: str | None = None
    usb_type: str
    physical_port: str
    production_profile_supported: bool
    ownership: OwnershipState
    owner_processes: list[str] = Field(default_factory=list)
    video_nodes: list[str] = Field(default_factory=list)


class DeviceProbe(Protocol):
    def enumerate(self) -> list[CameraObservation]: ...


class PreflightService:
    def __init__(self, *, registry: CandidateRegistry, probe: DeviceProbe):
        self._registry = registry
        self._probe = probe

    def run(self) -> PreflightReport:
        try:
            observations = self._probe.enumerate()
        except Exception as exc:
            return self._probe_failure(type(exc).__name__)

        by_serial = {observation.serial: observation for observation in observations}
        issues: list[PreflightIssue] = []
        cameras: list[PreflightCamera] = []
        for expected in self._registry.cameras:
            observed = by_serial.get(expected.serial)
            if observed is None:
                cameras.append(
                    PreflightCamera(
                        role=expected.role,
                        expected_serial=expected.serial,
                        connected=False,
                        identity_match=False,
                    )
                )
                issues.append(
                    PreflightIssue(
                        code="EXPECTED_CAMERA_MISSING",
                        severity=IssueSeverity.ERROR,
                        message=f"Expected camera {expected.role} was not enumerated",
                        camera_role=expected.role,
                    )
                )
                continue

            cameras.append(
                PreflightCamera(
                    role=expected.role,
                    expected_serial=expected.serial,
                    connected=True,
                    identity_match=observed.serial == expected.serial,
                    actual_serial=observed.serial,
                    product_name=observed.product_name,
                    firmware=observed.firmware,
                    recommended_firmware=observed.recommended_firmware,
                    usb_type=observed.usb_type,
                    physical_port=observed.physical_port,
                    production_profile_supported=observed.production_profile_supported,
                    ownership=observed.ownership,
                    owner_processes=observed.owner_processes,
                    video_nodes=observed.video_nodes,
                )
            )
            if not observed.production_profile_supported:
                issues.append(
                    PreflightIssue(
                        code="PRODUCTION_PROFILE_UNSUPPORTED",
                        severity=IssueSeverity.BLOCKING,
                        message=(
                            f"{expected.role} does not advertise "
                            f"{self._registry.profile.width}x{self._registry.profile.height}@{self._registry.profile.fps} "
                            f"{self._registry.profile.format}"
                        ),
                        camera_role=expected.role,
                    )
                )
            if observed.recommended_firmware and observed.firmware != observed.recommended_firmware:
                issues.append(
                    PreflightIssue(
                        code="FIRMWARE_DIFFERS_FROM_RECOMMENDED",
                        severity=IssueSeverity.WARNING,
                        message=(
                            f"{expected.role} firmware {observed.firmware} differs from "
                            f"recommended {observed.recommended_firmware}; no update was attempted"
                        ),
                        camera_role=expected.role,
                    )
                )
            if observed.ownership is OwnershipState.BUSY:
                issues.append(
                    PreflightIssue(
                        code="CAMERA_BUSY",
                        severity=IssueSeverity.BLOCKING,
                        message=f"{expected.role} is owned by another process",
                        camera_role=expected.role,
                    )
                )
            elif observed.ownership is OwnershipState.UNKNOWN:
                issues.append(
                    PreflightIssue(
                        code="CAMERA_OWNERSHIP_UNKNOWN",
                        severity=IssueSeverity.BLOCKING,
                        message=f"Exclusive ownership of {expected.role} could not be proven",
                        camera_role=expected.role,
                    )
                )

        status = self._status_for(issues)
        return PreflightReport(
            status=status,
            registry_source=self._registry.source_path,
            registry_sha256=self._registry.source_sha256,
            cameras=cameras,
            issues=issues,
        )

    def _probe_failure(self, error_name: str) -> PreflightReport:
        return PreflightReport(
            status=PreflightStatus.FAILED,
            registry_source=self._registry.source_path,
            registry_sha256=self._registry.source_sha256,
            cameras=[
                PreflightCamera(
                    role=expected.role,
                    expected_serial=expected.serial,
                    connected=False,
                    identity_match=False,
                )
                for expected in self._registry.cameras
            ],
            issues=[
                PreflightIssue(
                    code="DEVICE_PROBE_FAILED",
                    severity=IssueSeverity.ERROR,
                    message=f"Read-only device probe failed: {error_name}",
                )
            ],
        )

    @staticmethod
    def _status_for(issues: list[PreflightIssue]) -> PreflightStatus:
        if any(issue.severity is IssueSeverity.ERROR for issue in issues):
            return PreflightStatus.FAILED
        if any(issue.severity is IssueSeverity.BLOCKING for issue in issues):
            return PreflightStatus.BLOCKED
        return PreflightStatus.READY
