from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .models import CaptureStatus
from .models import CreateSessionRequest
from .models import IntrinsicsRoleRequest
from .models import PreflightReport
from .models import SampleRecord
from .models import SessionRecord
from .intrinsics_capture import RgbSnapshot
from .sessions import SessionNotFoundError
from .sessions import SessionStore
from .sessions import SessionTransitionError
from .workflow import BottleValidationResult
from .workflow import BottleFixtureContractRequest
from .workflow import FrozenBottleFixtureContract
from .workflow import BottleTagCaptureRequest
from .workflow import BottleTrialCaptureResult
from .workflow import BottleTrialObservation
from .workflow import CalibrationWorkflow
from .workflow import ExportBundleRequest
from .workflow import ExportResult
from .workflow import FactoryCameraSnapshot
from .workflow import FactorySnapshotBundle
from .workflow import FrozenTablePointContract
from .workflow import TableObservationsRequest
from .workflow import TablePointContractRequest
from .workflow import TableRegistrationResult
from .workflow import TransformRecord
from .workflow import WorkflowGateError
from .workflow import WorldOriginResult
from .workflow import WorldOriginCaptureBatch
from .workflow import WorldOriginPhysicalRequest
from .workflow import WorldOriginSolveRequest


class CaptureClient(Protocol):
    def run_preflight(self) -> PreflightReport: ...

    def snapshot_factory_intrinsics(self) -> list[FactoryCameraSnapshot]: ...

    def capture_world_origin(
        self,
        session_id: str,
        *,
        tag_size_m: float,
        tag_plane_height_m: float,
        frame_count: int,
    ) -> WorldOriginCaptureBatch: ...

    def capture_table_snapshot(self, session_id: str) -> RgbSnapshot: ...

    def start_intrinsics(self, session_id: str, role: str) -> CaptureStatus: ...

    def intrinsics_status(self) -> CaptureStatus: ...

    def preview_jpeg(self) -> bytes: ...

    def capture_sample(self) -> SampleRecord: ...

    def stop_intrinsics(self) -> CaptureStatus: ...


def create_orchestrator_app(
    capture_client: CaptureClient,
    store: SessionStore,
    workflow: CalibrationWorkflow | None = None,
) -> FastAPI:
    calibration = workflow or CalibrationWorkflow()
    app = FastAPI(title="ALOHA Calibration Orchestrator")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:4173", "http://localhost:4173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.get("/health")
    def health() -> dict[str, str | bool]:
        return {"service": "orchestrator", "status": "ok", "robot_command_api": False}

    @app.post("/api/sessions", response_model=SessionRecord, status_code=201)
    def create_session(request: CreateSessionRequest) -> SessionRecord:
        try:
            return store.create(request.name)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/sessions/{session_id}", response_model=SessionRecord)
    def get_session(session_id: str) -> SessionRecord:
        try:
            return store.get(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc

    @app.post("/api/sessions/{session_id}/actions/preflight", response_model=PreflightReport)
    def run_preflight(session_id: str) -> PreflightReport:
        try:
            store.get(session_id)
            report = capture_client.run_preflight()
            store.record_preflight(session_id, report)
            return report
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent preflight failed") from exc

    @app.post("/api/preflight-session", response_model=SessionRecord, status_code=201)
    def create_and_run_preflight() -> SessionRecord:
        record = store.create("camera-preflight")
        try:
            report = capture_client.run_preflight()
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent preflight failed") from exc
        return store.record_preflight(record.id, report)

    @app.post(
        "/api/sessions/{session_id}/actions/factory/freeze",
        response_model=FactorySnapshotBundle,
    )
    def freeze_factory_intrinsics(session_id: str) -> FactorySnapshotBundle:
        try:
            if store.get(session_id).state != "PREFLIGHT_READY":
                raise SessionTransitionError(
                    f"FACTORY_INTRINSICS_FROZEN requires PREFLIGHT_READY, current state is {store.get(session_id).state}"
                )
            bundle = FactorySnapshotBundle(cameras=capture_client.snapshot_factory_intrinsics())
            store.record_workflow_artifact(
                session_id,
                expected_state="PREFLIGHT_READY",
                next_state="FACTORY_INTRINSICS_FROZEN",
                artifact_name="factory_intrinsics.json",
                payload=bundle,
            )
            return bundle
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Factory intrinsics snapshot failed") from exc

    @app.post(
        "/api/sessions/{session_id}/actions/world-origin/solve",
        response_model=WorldOriginResult,
    )
    def solve_world_origin(session_id: str, request: WorldOriginSolveRequest) -> WorldOriginResult:
        try:
            if store.get(session_id).state != "FACTORY_INTRINSICS_FROZEN":
                raise SessionTransitionError(
                    "WORLD_ORIGIN_SOLVED requires FACTORY_INTRINSICS_FROZEN, "
                    f"current state is {store.get(session_id).state}"
                )
            result = calibration.solve_world_origin(
                samples=request.samples,
                world_from_tag=request.world_from_tag,
                total_frames=request.total_frames,
            )
            store.record_workflow_artifact(
                session_id,
                expected_state="FACTORY_INTRINSICS_FROZEN",
                next_state="WORLD_ORIGIN_SOLVED",
                artifact_name="world_origin.json",
                payload=result,
            )
            return result
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WorkflowGateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/api/sessions/{session_id}/actions/world-origin/capture-solve",
        response_model=WorldOriginResult,
    )
    def capture_and_solve_world_origin(
        session_id: str,
        request: WorldOriginPhysicalRequest,
    ) -> WorldOriginResult:
        try:
            if store.get(session_id).state != "FACTORY_INTRINSICS_FROZEN":
                raise SessionTransitionError(
                    "WORLD_ORIGIN_SOLVED requires FACTORY_INTRINSICS_FROZEN, "
                    f"current state is {store.get(session_id).state}"
                )
            batch = capture_client.capture_world_origin(
                session_id,
                tag_size_m=request.tag_size_m,
                tag_plane_height_m=request.tag_plane_height_m,
                frame_count=request.frame_count,
            )
            result = calibration.solve_world_origin(
                samples=batch.samples,
                world_from_tag=batch.world_from_tag,
                total_frames=batch.total_frames,
            )
            store.record_workflow_artifact(
                session_id,
                expected_state="FACTORY_INTRINSICS_FROZEN",
                next_state="WORLD_ORIGIN_SOLVED",
                artifact_name="world_origin.json",
                payload=result,
            )
            return result
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WorkflowGateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="World-origin capture failed") from exc

    @app.post(
        "/api/sessions/{session_id}/actions/table-contract/freeze",
        response_model=FrozenTablePointContract,
    )
    def freeze_table_contract(
        session_id: str,
        request: TablePointContractRequest,
    ) -> FrozenTablePointContract:
        try:
            if store.get(session_id).state != "WORLD_ORIGIN_SOLVED":
                raise SessionTransitionError(
                    "TABLE_POINT_CONTRACT_FROZEN requires WORLD_ORIGIN_SOLVED, "
                    f"current state is {store.get(session_id).state}"
                )
            contract = calibration.freeze_table_point_contract(request)
            store.record_workflow_artifact(
                session_id,
                expected_state="WORLD_ORIGIN_SOLVED",
                next_state="TABLE_POINT_CONTRACT_FROZEN",
                artifact_name="table_point_contract.json",
                payload=contract,
            )
            return contract
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WorkflowGateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/actions/table/snapshot")
    def capture_table_snapshot(session_id: str) -> Response:
        try:
            state = store.get(session_id).state
            if state not in {"WORLD_ORIGIN_SOLVED", "TABLE_POINT_CONTRACT_FROZEN"}:
                raise SessionTransitionError(
                    "table snapshot requires WORLD_ORIGIN_SOLVED or TABLE_POINT_CONTRACT_FROZEN, "
                    f"current state is {state}"
                )
            snapshot = capture_client.capture_table_snapshot(session_id)
            return Response(
                content=snapshot.jpeg,
                media_type="image/jpeg",
                headers={
                    "Cache-Control": "no-store",
                    "X-Attempt-Id": snapshot.attempt_id,
                    "X-Frame-Number": str(snapshot.frame_number),
                    "X-Device-Timestamp-Ms": str(snapshot.device_timestamp_ms),
                    "X-Image-Sha256": snapshot.image_sha256,
                },
            )
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Table snapshot capture failed") from exc

    @app.post(
        "/api/sessions/{session_id}/actions/table/solve",
        response_model=TableRegistrationResult,
    )
    def solve_table_registration(
        session_id: str,
        request: TableObservationsRequest,
    ) -> TableRegistrationResult:
        try:
            if store.get(session_id).state != "TABLE_POINT_CONTRACT_FROZEN":
                raise SessionTransitionError(
                    "WORLD_REGISTRATION_VALIDATED requires TABLE_POINT_CONTRACT_FROZEN, "
                    f"current state is {store.get(session_id).state}"
                )
            bundle = FactorySnapshotBundle.model_validate(
                store.read_workflow_artifact(session_id, "factory_intrinsics.json")
            )
            intrinsics = next(camera.intrinsics for camera in bundle.cameras if camera.role == "cam_high")
            anchor = WorldOriginResult.model_validate(
                store.read_workflow_artifact(session_id, "world_origin.json")
            )
            contract = FrozenTablePointContract.model_validate(
                store.read_workflow_artifact(session_id, "table_point_contract.json")
            )
            result = calibration.solve_table_registration(
                points=calibration.observations_from_contract(
                    contract=contract,
                    observations=request.observations,
                ),
                intrinsics=intrinsics,
                initial_world_from_camera=anchor.world_from_camera,
            )
            store.record_workflow_artifact(
                session_id,
                expected_state="TABLE_POINT_CONTRACT_FROZEN",
                next_state="WORLD_REGISTRATION_VALIDATED",
                artifact_name="table_registration.json",
                payload=result,
            )
            return result
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WorkflowGateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/api/sessions/{session_id}/actions/bottle-contract/freeze",
        response_model=FrozenBottleFixtureContract,
    )
    def freeze_bottle_fixture_contract(
        session_id: str,
        request: BottleFixtureContractRequest,
    ) -> FrozenBottleFixtureContract:
        try:
            if store.get(session_id).state != "WORLD_REGISTRATION_VALIDATED":
                raise SessionTransitionError(
                    "BOTTLE_FIXTURE_CONTRACT_FROZEN requires WORLD_REGISTRATION_VALIDATED, "
                    f"current state is {store.get(session_id).state}"
                )
            contract = calibration.freeze_bottle_fixture_contract(request)
            store.record_workflow_artifact(
                session_id,
                expected_state="WORLD_REGISTRATION_VALIDATED",
                next_state="BOTTLE_FIXTURE_CONTRACT_FROZEN",
                artifact_name="bottle_fixture_contract.json",
                payload=contract,
            )
            return contract
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WorkflowGateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/api/sessions/{session_id}/actions/bottle/{trial_id}/capture",
        response_model=BottleTrialCaptureResult,
    )
    def capture_bottle_trial(
        session_id: str,
        trial_id: str,
        request: BottleTagCaptureRequest,
    ) -> BottleTrialCaptureResult:
        try:
            if trial_id not in {"B-A", "B-B", "B-C"}:
                raise WorkflowGateError("trial_id must be B-A, B-B, or B-C")
            if store.get(session_id).state != "BOTTLE_FIXTURE_CONTRACT_FROZEN":
                raise SessionTransitionError(
                    "bottle capture requires BOTTLE_FIXTURE_CONTRACT_FROZEN, "
                    f"current state is {store.get(session_id).state}"
                )
            batch = capture_client.capture_world_origin(
                session_id,
                tag_size_m=request.tag_size_m,
                tag_plane_height_m=0.0,
                frame_count=request.frame_count,
            )
            stability = calibration.aggregate_tag_pose(
                samples=batch.samples,
                minimum_frames=150,
                total_frames=batch.total_frames,
            )
            result = BottleTrialCaptureResult(
                observation=BottleTrialObservation(
                    id=trial_id,
                    camera_from_tag=stability.camera_from_tag,
                ),
                stability=stability,
            )
            store.write_numbered_attempt(
                session_id,
                f"bottle_{trial_id.lower().replace('-', '_')}",
                result,
            )
            return result
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WorkflowGateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Bottle trial capture failed") from exc

    @app.post(
        "/api/sessions/{session_id}/actions/bottle/validate",
        response_model=BottleValidationResult,
    )
    def validate_bottle_trials(session_id: str) -> BottleValidationResult:
        try:
            if store.get(session_id).state != "BOTTLE_FIXTURE_CONTRACT_FROZEN":
                raise SessionTransitionError(
                    "TAGGED_FIXTURE_TRANSFER_PASS requires BOTTLE_FIXTURE_CONTRACT_FROZEN, "
                    f"current state is {store.get(session_id).state}"
                )
            table = TableRegistrationResult.model_validate(
                store.read_workflow_artifact(session_id, "table_registration.json")
            )
            table_contract = FrozenTablePointContract.model_validate(
                store.read_workflow_artifact(session_id, "table_point_contract.json")
            )
            fixture = FrozenBottleFixtureContract.model_validate(
                store.read_workflow_artifact(session_id, "bottle_fixture_contract.json")
            )
            observations = [
                BottleTrialCaptureResult.model_validate(
                    store.read_latest_attempt(
                        session_id,
                        f"bottle_{trial_id.lower().replace('-', '_')}",
                    )
                ).observation
                for trial_id in ("B-A", "B-B", "B-C")
            ]
            result = calibration.validate_bottle_observations(
                world_from_camera=table.world_from_camera,
                fixture=fixture,
                table_contract=table_contract,
                observations=observations,
            )
            store.record_workflow_artifact(
                session_id,
                expected_state="BOTTLE_FIXTURE_CONTRACT_FROZEN",
                next_state="TAGGED_FIXTURE_TRANSFER_PASS",
                artifact_name="bottle_validation.json",
                payload=result,
            )
            return result
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WorkflowGateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/api/sessions/{session_id}/actions/export",
        response_model=ExportResult,
    )
    def export_calibration(session_id: str, request: ExportBundleRequest) -> ExportResult:
        try:
            if store.get(session_id).state != "TAGGED_FIXTURE_TRANSFER_PASS":
                raise SessionTransitionError(
                    "EXPORT_READY requires TAGGED_FIXTURE_TRANSFER_PASS, "
                    f"current state is {store.get(session_id).state}"
                )
            table = TableRegistrationResult.model_validate(
                store.read_workflow_artifact(session_id, "table_registration.json")
            )
            bottle_validation = BottleValidationResult.model_validate(
                store.read_workflow_artifact(session_id, "bottle_validation.json")
            )
            fixture = FrozenBottleFixtureContract.model_validate(
                store.read_workflow_artifact(session_id, "bottle_fixture_contract.json")
            )
            result = calibration.export_calibration_bundle(
                output_dir=store.export_output_dir(session_id),
                stage=request.stage,
                world_from_camera=table.world_from_camera,
                bottle_asset_path=request.bottle_asset_path,
                bottle_asset_sha256=request.bottle_asset_sha256,
                bottle_asset_prim=request.bottle_asset_prim,
                bottle_validation=bottle_validation,
                task_from_asset=fixture.task_from_asset,
            )
            store.record_workflow_artifact(
                session_id,
                expected_state="TAGGED_FIXTURE_TRANSFER_PASS",
                next_state="EXPORT_READY",
                artifact_name="export_manifest.json",
                payload=result,
            )
            return result
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except WorkflowGateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/sessions/{session_id}/actions/intrinsics/start", response_model=CaptureStatus)
    def start_intrinsics(session_id: str, request: IntrinsicsRoleRequest) -> CaptureStatus:
        try:
            store.assert_intrinsics_start_allowed(session_id)
            status = capture_client.start_intrinsics(session_id, request.role)
            try:
                store.record_intrinsics_start(session_id, status)
            except Exception:
                capture_client.stop_intrinsics()
                raise
            return status
        except SessionNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Calibration session not found") from exc
        except SessionTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent intrinsics start failed") from exc

    @app.get("/api/intrinsics/status", response_model=CaptureStatus)
    def intrinsics_status() -> CaptureStatus:
        try:
            return capture_client.intrinsics_status()
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent status failed") from exc

    @app.get("/api/intrinsics/preview.jpg")
    def intrinsics_preview() -> Response:
        try:
            return Response(
                content=capture_client.preview_jpeg(),
                media_type="image/jpeg",
                headers={"Cache-Control": "no-store"},
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent preview failed") from exc

    @app.post("/api/intrinsics/sample", response_model=SampleRecord)
    def capture_sample() -> SampleRecord:
        try:
            return capture_client.capture_sample()
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent sample failed") from exc

    @app.post("/api/intrinsics/stop", response_model=CaptureStatus)
    def stop_intrinsics() -> CaptureStatus:
        try:
            return capture_client.stop_intrinsics()
        except Exception as exc:
            raise HTTPException(status_code=502, detail="Capture agent stop failed") from exc

    return app
