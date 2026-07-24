"""Fenced leader/follower execution on one legacy Cloud TPU v6e-64 slice.

Legacy Cloud TPU starts the same program on sixteen VMs.  Process zero alone
owns ``RuntimeSupervisorV1`` and its mutable lease; processes 1..15 execute a
fixed follower callback.  This adapter uses JAX's already-running distributed
coordination service, so an accelerator step costs no GCS operations and needs
no second socket service:

1. process zero publishes one write-once, attempt-bound permit;
2. all sixteen processes cross a native READY barrier;
3. all execute the same fixed accelerator callback; and
4. all cross a native COMPLETED barrier before a checkpoint may be published.

Every wait also verifies the original sixteen process incarnations.  A missing
or restarted member closes the supervisor's one-way gate and raises
``SliceMemberLost``.  The command contains only a permit or terminal outcome;
metadata never supplies a callable, path, or model option.

JAX 0.10.2 does not publicly expose the coordination client used by its own
multihost helpers.  The worker lock pins that exact version, and the narrow
adapter below rejects any other version or method surface.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, Self

from .preemption_v2 import AttemptAuthorityV2
from .protocol import (
    IDENTIFIER_PATTERN,
    SHA256_PATTERN,
    CanonicalRecord,
    ProtocolError,
    canonical_sha256,
    parse_json,
    require_integer,
    require_record_object,
    require_string,
)
from .runtime_supervisor import (
    ResolvedRuntimeLaunchV1,
    RuntimeLaunchAdmissionV1,
    SliceMemberLost,
)

LEGACY_PROCESS_COUNT = 16
LEGACY_LOCAL_DEVICE_COUNT = 4
LEGACY_DEVICE_COUNT = 64
PINNED_JAX_VERSION = "0.10.2"
_ALL_PROCESS_IDS = tuple(range(LEGACY_PROCESS_COUNT))
_OUTCOMES = frozenset({"SUCCEEDED", "PREEMPTED", "FAILED"})
_COMMAND_KINDS = frozenset({"STEP", "TERMINATE"})
_MAX_COMMAND_BYTES = 4096


class LegacySliceRuntimeError(RuntimeError):
    """The fixed legacy whole-slice boundary failed closed."""


class LegacyCoordinationError(LegacySliceRuntimeError):
    """The pinned JAX coordinator was unavailable or inconsistent."""


def jax_block_until_ready_pytree[T](value: T) -> T:
    """Synchronize every device-backed leaf produced by an accelerator step."""

    import jax

    if jax.__version__ != PINNED_JAX_VERSION:
        raise LegacySliceRuntimeError(
            "device completion requires the pinned JAX runtime"
        )
    jax.block_until_ready(value)
    return value


def _sha256(value: Any, name: str) -> str:
    return require_string(value, name, pattern=SHA256_PATTERN)


@dataclass(frozen=True, slots=True)
class LegacySliceBindingV1(CanonicalRecord):
    """Pre-lease identity independently resolved by every slice process."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.legacy-slice-binding.v1"

    launch_token_sha256: str
    launch_admission_sha256: str
    deployment_envelope_sha256: str
    implementation_identity_sha256: str
    attempt_id: str
    expected_fence: int
    maximum_step_permits: int
    process_count: int
    device_count: int

    def __post_init__(self) -> None:
        _sha256(self.launch_token_sha256, "launch_token_sha256")
        _sha256(self.launch_admission_sha256, "launch_admission_sha256")
        _sha256(
            self.deployment_envelope_sha256,
            "deployment_envelope_sha256",
        )
        _sha256(
            self.implementation_identity_sha256,
            "implementation_identity_sha256",
        )
        require_string(
            self.attempt_id,
            "attempt_id",
            pattern=IDENTIFIER_PATTERN,
        )
        require_integer(self.expected_fence, "expected_fence", minimum=1)
        require_integer(
            self.maximum_step_permits,
            "maximum_step_permits",
            minimum=1,
            maximum=1_000_000,
        )
        if (
            require_integer(self.process_count, "process_count", minimum=1)
            != LEGACY_PROCESS_COUNT
        ):
            raise ProtocolError("legacy slice requires exactly 16 processes")
        if (
            require_integer(self.device_count, "device_count", minimum=1)
            != LEGACY_DEVICE_COUNT
        ):
            raise ProtocolError("legacy slice requires exactly 64 devices")

    @classmethod
    def from_resolved(
        cls,
        resolved: ResolvedRuntimeLaunchV1,
        *,
        maximum_step_permits: int,
    ) -> LegacySliceBindingV1:
        if type(resolved) is not ResolvedRuntimeLaunchV1:
            raise TypeError("resolved must be ResolvedRuntimeLaunchV1")
        token = resolved.launch_token
        admission = resolved.admission
        if resolved.launch_ref.sha256 != token.digest:
            raise ProtocolError("launch reference does not bind the token")
        if resolved.admission_version.sha256 != admission.digest:
            raise ProtocolError("admission version changed identity")
        return cls(
            launch_token_sha256=token.digest,
            launch_admission_sha256=admission.digest,
            deployment_envelope_sha256=admission.deployment.digest,
            implementation_identity_sha256=(
                token.implementation_identity_sha256
            ),
            attempt_id=token.attempt_id,
            expected_fence=token.expected_fence,
            maximum_step_permits=maximum_step_permits,
            process_count=LEGACY_PROCESS_COUNT,
            device_count=LEGACY_DEVICE_COUNT,
        )

    def verify_admission(self, admission: RuntimeLaunchAdmissionV1) -> None:
        if type(admission) is not RuntimeLaunchAdmissionV1:
            raise TypeError("admission must be RuntimeLaunchAdmissionV1")
        deployment = admission.deployment
        if (
            admission.digest != self.launch_admission_sha256
            or deployment.digest != self.deployment_envelope_sha256
            or deployment.slice_attempt_id != self.attempt_id
            or deployment.vm_count != LEGACY_PROCESS_COUNT
            or deployment.chip_count != LEGACY_DEVICE_COUNT
            or deployment.accelerator_type != "v6e-64"
            or deployment.topology != "8x8"
        ):
            raise LegacySliceRuntimeError(
                "admission differs from the fixed legacy slice binding"
            )

    def verify_authority(self, authority: AttemptAuthorityV2) -> None:
        if type(authority) is not AttemptAuthorityV2:
            raise TypeError("authority must be AttemptAuthorityV2")
        if (
            authority.attempt_id != self.attempt_id
            or authority.fence != self.expected_fence
            or authority.deployment.digest
            != self.deployment_envelope_sha256
            or authority.deployment_identity.implementation_identity_sha256
            != self.implementation_identity_sha256
        ):
            raise SliceMemberLost(
                "leader authority differs from the immutable slice binding"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "launch_token_sha256": self.launch_token_sha256,
            "launch_admission_sha256": self.launch_admission_sha256,
            "deployment_envelope_sha256": self.deployment_envelope_sha256,
            "implementation_identity_sha256": (
                self.implementation_identity_sha256
            ),
            "attempt_id": self.attempt_id,
            "expected_fence": self.expected_fence,
            "maximum_step_permits": self.maximum_step_permits,
            "process_count": self.process_count,
            "device_count": self.device_count,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        fields = (
            "launch_token_sha256",
            "launch_admission_sha256",
            "deployment_envelope_sha256",
            "implementation_identity_sha256",
            "attempt_id",
            "expected_fence",
            "maximum_step_permits",
            "process_count",
            "device_count",
        )
        obj = require_record_object(value, schema=cls.SCHEMA, fields=fields)
        return cls(
            launch_token_sha256=_sha256(
                obj["launch_token_sha256"], "launch_token_sha256"
            ),
            launch_admission_sha256=_sha256(
                obj["launch_admission_sha256"],
                "launch_admission_sha256",
            ),
            deployment_envelope_sha256=_sha256(
                obj["deployment_envelope_sha256"],
                "deployment_envelope_sha256",
            ),
            implementation_identity_sha256=_sha256(
                obj["implementation_identity_sha256"],
                "implementation_identity_sha256",
            ),
            attempt_id=require_string(obj["attempt_id"], "attempt_id"),
            expected_fence=require_integer(
                obj["expected_fence"], "expected_fence", minimum=1
            ),
            maximum_step_permits=require_integer(
                obj["maximum_step_permits"],
                "maximum_step_permits",
                minimum=1,
                maximum=1_000_000,
            ),
            process_count=require_integer(
                obj["process_count"], "process_count", minimum=1
            ),
            device_count=require_integer(
                obj["device_count"], "device_count", minimum=1
            ),
        )


@dataclass(frozen=True, slots=True)
class LegacySliceCommandV1(CanonicalRecord):
    """One write-once permit or terminal command with no executable payload."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.legacy-slice-command.v1"

    slice_binding_sha256: str
    sequence: int
    kind: str
    permit: int | None
    outcome: str | None

    def __post_init__(self) -> None:
        _sha256(self.slice_binding_sha256, "slice_binding_sha256")
        sequence = require_integer(self.sequence, "sequence", minimum=1)
        if self.kind not in _COMMAND_KINDS:
            raise ProtocolError("legacy slice command kind is invalid")
        if self.kind == "STEP":
            if (
                type(self.permit) is not int
                or self.permit != sequence
                or self.outcome is not None
            ):
                raise ProtocolError(
                    "step command requires permit=sequence and no outcome"
                )
        elif self.permit is not None or self.outcome not in _OUTCOMES:
            raise ProtocolError(
                "terminal command requires no permit and one fixed outcome"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "slice_binding_sha256": self.slice_binding_sha256,
            "sequence": self.sequence,
            "kind": self.kind,
            "permit": self.permit,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = require_record_object(
            value,
            schema=cls.SCHEMA,
            fields=(
                "slice_binding_sha256",
                "sequence",
                "kind",
                "permit",
                "outcome",
            ),
        )
        permit = obj["permit"]
        outcome = obj["outcome"]
        return cls(
            slice_binding_sha256=_sha256(
                obj["slice_binding_sha256"], "slice_binding_sha256"
            ),
            sequence=require_integer(obj["sequence"], "sequence", minimum=1),
            kind=require_string(obj["kind"], "kind"),
            permit=(
                None
                if permit is None
                else require_integer(permit, "permit", minimum=1)
            ),
            outcome=(
                None if outcome is None else require_string(outcome, "outcome")
            ),
        )


class LegacyCoordinationClient(Protocol):
    """Only the pinned JAX coordination operations the adapter requires."""

    @property
    def production_qualified(self) -> bool: ...

    @property
    def process_index(self) -> int: ...

    @property
    def process_count(self) -> int: ...

    def put_once(self, key: str, value: bytes) -> None: ...

    def wait_get(self, key: str, *, timeout_seconds: float) -> bytes: ...

    def barrier(
        self,
        barrier_id: str,
        *,
        timeout_seconds: float,
        process_ids: Sequence[int],
    ) -> None: ...

    def live_nodes(self, process_ids: Sequence[int]) -> Mapping[int, int]: ...


class JAXCoordinationClientV1:
    """Exact JAX 0.10.2 coordination client used by legacy TPU discovery."""

    def __init__(self) -> None:
        try:
            import jax
            from jax._src import distributed
        except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover
            raise LegacyCoordinationError("pinned JAX is unavailable") from error
        if jax.__version__ != PINNED_JAX_VERSION:
            raise LegacyCoordinationError("JAX coordination version changed")
        client = distributed.global_state.client
        if client is None or not jax.distributed.is_initialized():
            raise LegacyCoordinationError(
                "JAX distributed runtime is not initialized"
            )
        required = (
            "get_live_nodes",
            "blocking_key_value_get_bytes",
            "key_value_set_bytes",
            "wait_at_barrier",
        )
        if any(not callable(getattr(client, name, None)) for name in required):
            raise LegacyCoordinationError(
                "pinned JAX coordination API is incomplete"
            )
        process_index = jax.process_index()
        process_count = jax.process_count()
        if (
            type(process_index) is not int
            or process_count != LEGACY_PROCESS_COUNT
            or not 0 <= process_index < LEGACY_PROCESS_COUNT
        ):
            raise LegacyCoordinationError(
                "legacy whole slice requires JAX processes 0..15"
            )
        self._client = client
        self._process_index = process_index

    @property
    def production_qualified(self) -> bool:
        return True

    @property
    def process_index(self) -> int:
        return self._process_index

    @property
    def process_count(self) -> int:
        return LEGACY_PROCESS_COUNT

    def put_once(self, key: str, value: bytes) -> None:
        try:
            self._client.key_value_set_bytes(
                key,
                value,
                allow_overwrite=False,
            )
        except Exception as error:
            raise LegacyCoordinationError(
                "write-once JAX coordination publication failed"
            ) from error

    def wait_get(self, key: str, *, timeout_seconds: float) -> bytes:
        timeout_ms = int(timeout_seconds * 1000)
        try:
            value = self._client.blocking_key_value_get_bytes(
                key,
                timeout_ms,
            )
        except Exception as error:
            raise LegacyCoordinationError(
                "blocking JAX coordination read failed"
            ) from error
        if type(value) is not bytes:
            raise LegacyCoordinationError(
                "JAX coordination read returned non-bytes"
            )
        return value

    def barrier(
        self,
        barrier_id: str,
        *,
        timeout_seconds: float,
        process_ids: Sequence[int],
    ) -> None:
        timeout_ms = int(timeout_seconds * 1000)
        try:
            self._client.wait_at_barrier(
                barrier_id,
                timeout_ms,
                process_ids=list(process_ids),
            )
        except Exception as error:
            raise LegacyCoordinationError(
                "JAX whole-slice barrier failed"
            ) from error

    def live_nodes(self, process_ids: Sequence[int]) -> Mapping[int, int]:
        try:
            observed = self._client.get_live_nodes(list(process_ids))
        except Exception as error:
            raise LegacyCoordinationError(
                "JAX liveness read failed"
            ) from error
        if not isinstance(observed, Mapping):
            raise LegacyCoordinationError("JAX liveness response is malformed")
        result: dict[int, int] = {}
        for key, value in observed.items():
            if type(key) is not int or type(value) is not int or value < 0:
                raise LegacyCoordinationError(
                    "JAX liveness response has invalid identities"
                )
            result[key] = value
        return result


def jax_device_inventory_sha256(binding: LegacySliceBindingV1) -> str:
    """Hash the exact 64-device enumeration after strict v6e validation."""

    if type(binding) is not LegacySliceBindingV1:
        raise TypeError("binding must be LegacySliceBindingV1")
    try:
        import jax
    except (ImportError, ModuleNotFoundError) as error:  # pragma: no cover
        raise LegacySliceRuntimeError("JAX is unavailable") from error
    devices = tuple(jax.devices())
    if (
        jax.process_count() != LEGACY_PROCESS_COUNT
        or jax.local_device_count() != LEGACY_LOCAL_DEVICE_COUNT
        or len(devices) != LEGACY_DEVICE_COUNT
    ):
        raise LegacySliceRuntimeError(
            "runtime is not one exact 16-process v6e-64 slice"
        )
    records: list[dict[str, Any]] = []
    owners: list[int] = []
    device_ids: set[int] = set()
    coordinate_cores: set[tuple[tuple[int, ...], int]] = set()
    for device in devices:
        try:
            device_id = int(device.id)
            process_index = int(device.process_index)
            coordinates = tuple(int(value) for value in device.coords)
            core_on_chip = int(device.core_on_chip)
            platform = str(device.platform)
            device_kind = str(device.device_kind)
        except (AttributeError, TypeError, ValueError) as error:
            raise LegacySliceRuntimeError(
                "JAX device identity is incomplete"
            ) from error
        if (
            device_id in device_ids
            or not 0 <= process_index < LEGACY_PROCESS_COUNT
            or len(coordinates) != 3
            or (coordinates, core_on_chip) in coordinate_cores
            or platform != "tpu"
            or "v6" not in device_kind.lower()
        ):
            raise LegacySliceRuntimeError(
                "JAX device inventory is not a unique v6e topology"
            )
        device_ids.add(device_id)
        coordinate_cores.add((coordinates, core_on_chip))
        owners.append(process_index)
        records.append(
            {
                "device_id": device_id,
                "process_index": process_index,
                "coordinates": list(coordinates),
                "core_on_chip": core_on_chip,
                "platform": platform,
                "device_kind": device_kind,
            }
        )
    if Counter(owners) != Counter(
        {
            process_index: LEGACY_LOCAL_DEVICE_COUNT
            for process_index in _ALL_PROCESS_IDS
        }
    ):
        raise LegacySliceRuntimeError(
            "device ownership is not four devices per process"
        )
    records.sort(key=lambda item: item["device_id"])
    return canonical_sha256(
        {
            "schema": "moe-sae.r1.legacy-jax-device-inventory.v1",
            "slice_binding_sha256": binding.digest,
            "process_count": LEGACY_PROCESS_COUNT,
            "local_device_count": LEGACY_LOCAL_DEVICE_COUNT,
            "device_count": LEGACY_DEVICE_COUNT,
            "devices": records,
        }
    )


class LegacyV6e64WholeSliceRuntimeV1:
    """One process-zero permit publisher plus fifteen deterministic followers."""

    def __init__(
        self,
        *,
        binding: LegacySliceBindingV1,
        coordination: LegacyCoordinationClient,
        inventory_sha256: Callable[[LegacySliceBindingV1], str] = (
            jax_device_inventory_sha256
        ),
        block_until_ready: Callable[[Any], Any] = (
            jax_block_until_ready_pytree
        ),
        poll_interval_seconds: float = 0.25,
        command_timeout_seconds: float = 900.0,
        barrier_timeout_seconds: float = 1800.0,
    ) -> None:
        if type(binding) is not LegacySliceBindingV1:
            raise TypeError("binding must be LegacySliceBindingV1")
        if coordination.production_qualified is not True:
            raise LegacySliceRuntimeError(
                "coordination client is not production-qualified"
            )
        if (
            type(coordination.process_count) is not int
            or type(coordination.process_index) is not int
            or coordination.process_count != LEGACY_PROCESS_COUNT
            or not 0 <= coordination.process_index < LEGACY_PROCESS_COUNT
        ):
            raise LegacySliceRuntimeError(
                "coordination client does not own one process in 0..15"
            )
        if not callable(inventory_sha256):
            raise TypeError("inventory_sha256 must be callable")
        if not callable(block_until_ready):
            raise TypeError("block_until_ready must be callable")
        for value, name, lower, upper in (
            (poll_interval_seconds, "poll_interval_seconds", 0.01, 5.0),
            (
                command_timeout_seconds,
                "command_timeout_seconds",
                1.0,
                3600.0,
            ),
            (
                barrier_timeout_seconds,
                "barrier_timeout_seconds",
                1.0,
                3600.0,
            ),
        ):
            if (
                type(value) not in {int, float}
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"{name} must be in [{lower}, {upper}]")
        self._binding = binding
        self._coordination = coordination
        self._inventory_sha256 = inventory_sha256
        self._block_until_ready = block_until_ready
        self._poll_seconds = float(poll_interval_seconds)
        self._command_timeout = float(command_timeout_seconds)
        self._barrier_timeout = float(barrier_timeout_seconds)
        self._root = f"moe-sae-r1/legacy-slice/{binding.digest}"
        self._incarnations: dict[int, int] | None = None
        self._last_published_sequence = 0
        self._last_completed_permit = 0
        self._initialized = False
        self._terminated = False
        self._stop_request: Callable[[str], None] | None = None
        self._member_loss = threading.Event()
        self._monitor_stop = threading.Event()
        self._monitor: threading.Thread | None = None
        self._lock = threading.RLock()

    @property
    def production_qualified(self) -> bool:
        return True

    @property
    def process_index(self) -> int:
        return self._coordination.process_index

    @property
    def is_leader(self) -> bool:
        return self.process_index == 0

    def bind_stop_request(
        self,
        request_stop: Callable[[str], None],
    ) -> None:
        if not callable(request_stop):
            raise TypeError("request_stop must be callable")
        with self._lock:
            if self._stop_request is not None:
                raise LegacySliceRuntimeError(
                    "member-loss callback is already bound"
                )
            self._stop_request = request_stop

    def initialize_device_inventory_sha256(
        self,
        admission: RuntimeLaunchAdmissionV1,
    ) -> str:
        self._binding.verify_admission(admission)
        with self._lock:
            if self._initialized:
                raise LegacySliceRuntimeError("runtime was initialized twice")
            if self.is_leader and self._stop_request is None:
                raise LegacySliceRuntimeError(
                    "leader member-loss callback is not bound"
                )
            inventory = _sha256(
                self._inventory_sha256(self._binding),
                "device_inventory_sha256",
            )
            self._incarnations = self._read_exact_incarnations()
            if self.is_leader:
                monitor = threading.Thread(
                    target=self._monitor_members,
                    name="r1-legacy-slice-liveness",
                    daemon=True,
                )
                monitor.start()
                self._monitor = monitor
            self._initialized = True
            return inventory

    def _read_exact_incarnations(self) -> dict[int, int]:
        try:
            observed = dict(
                self._coordination.live_nodes(_ALL_PROCESS_IDS)
            )
        except Exception as error:
            raise SliceMemberLost(
                "whole-slice liveness is unavailable"
            ) from error
        if set(observed) != set(_ALL_PROCESS_IDS) or any(
            type(value) is not int or value < 0 for value in observed.values()
        ):
            raise SliceMemberLost(
                "whole-slice liveness does not contain processes 0..15"
            )
        return observed

    def _assert_members_unchanged(self) -> None:
        if self._incarnations is None:
            raise LegacySliceRuntimeError("device inventory is not initialized")
        if self._read_exact_incarnations() != self._incarnations:
            raise SliceMemberLost(
                "a JAX process disappeared or changed incarnation"
            )

    def _latch_member_loss(self) -> None:
        with self._lock:
            if self._member_loss.is_set():
                return
            self._member_loss.set()
            callback = self._stop_request
        if callback is not None:
            callback("SLICE_MEMBER_LOST")

    def _monitor_members(self) -> None:
        while not self._monitor_stop.wait(self._poll_seconds):
            try:
                self._assert_members_unchanged()
            except (LegacySliceRuntimeError, SliceMemberLost):
                self._latch_member_loss()
                return

    def _require_open(self) -> None:
        if not self._initialized:
            raise LegacySliceRuntimeError("runtime is not initialized")
        if self._terminated:
            raise SliceMemberLost("whole-slice runtime is terminal")
        if self._member_loss.is_set():
            raise SliceMemberLost("whole-slice member loss is latched")

    def _command_key(self, sequence: int) -> str:
        return f"{self._root}/commands/{sequence:020d}"

    def _barrier_id(self, sequence: int, phase: str) -> str:
        return (
            f"moe-sae-r1-legacy-{self._binding.digest}-"
            f"{sequence:020d}-{phase}"
        )

    @staticmethod
    def _decode_command(raw: bytes) -> LegacySliceCommandV1:
        if type(raw) is not bytes or len(raw) > _MAX_COMMAND_BYTES:
            raise SliceMemberLost("slice command is not bounded bytes")
        try:
            command = LegacySliceCommandV1.from_dict(parse_json(raw))
        except (ProtocolError, ValueError) as error:
            raise SliceMemberLost("slice command is malformed") from error
        if command.canonical_bytes() != raw:
            raise SliceMemberLost("slice command is not canonical")
        return command

    def _publish_command(self, command: LegacySliceCommandV1) -> None:
        raw = command.canonical_bytes()
        if len(raw) > _MAX_COMMAND_BYTES:
            raise LegacySliceRuntimeError("slice command is oversized")
        try:
            self._coordination.put_once(
                self._command_key(command.sequence),
                raw,
            )
        except Exception as error:
            raise SliceMemberLost(
                "write-once slice command publication failed"
            ) from error

    def _wait_command(self, sequence: int) -> LegacySliceCommandV1:
        key = self._command_key(sequence)
        self._assert_members_unchanged()
        try:
            raw = self._coordination.wait_get(
                key,
                timeout_seconds=self._command_timeout,
            )
        except Exception as error:
            raise SliceMemberLost(
                "slice command read became ambiguous"
            ) from error
        self._assert_members_unchanged()
        command = self._decode_command(raw)
        if (
            command.slice_binding_sha256 != self._binding.digest
            or command.sequence != sequence
            or (
                command.kind == "STEP"
                and command.sequence
                > self._binding.maximum_step_permits
            )
            or (
                command.kind == "TERMINATE"
                and command.sequence
                > self._binding.maximum_step_permits + 1
            )
        ):
            raise SliceMemberLost(
                "slice command changed binding, sequence, or bound"
            )
        return command

    def _barrier(self, sequence: int, phase: str) -> None:
        try:
            self._coordination.barrier(
                self._barrier_id(sequence, phase),
                timeout_seconds=self._barrier_timeout,
                process_ids=_ALL_PROCESS_IDS,
            )
        except Exception as error:
            raise SliceMemberLost(
                f"whole-slice {phase} barrier failed"
            ) from error
        if self.is_leader and phase == "completed":
            self._assert_members_unchanged()

    def run_authorized_step[T](
        self,
        *,
        permit: int,
        authority: AttemptAuthorityV2,
        accelerator_step: Callable[[], T],
    ) -> T:
        """Publish one permit and return only after all 16 callbacks complete."""

        if not self.is_leader:
            raise LegacySliceRuntimeError(
                "only process zero may publish a step permit"
            )
        if not callable(accelerator_step):
            raise TypeError("accelerator_step must be callable")
        try:
            self._require_open()
            self._binding.verify_authority(authority)
            if (
                type(permit) is not int
                or permit != self._last_published_sequence + 1
                or permit > self._binding.maximum_step_permits
            ):
                raise SliceMemberLost("slice permit is not exactly monotonic")
            command = LegacySliceCommandV1(
                slice_binding_sha256=self._binding.digest,
                sequence=permit,
                kind="STEP",
                permit=permit,
                outcome=None,
            )
            self._assert_members_unchanged()
            self._publish_command(command)
            self._last_published_sequence = permit
            self._barrier(permit, "ready")
            try:
                result = accelerator_step()
                self._block_until_ready(result)
            except BaseException:
                # The STEP command is already visible. Release healthy
                # followers from this permit before the supervisor publishes
                # the next-sequence terminal command.
                self._barrier(permit, "completed")
                raise
            self._barrier(permit, "completed")
            self._last_completed_permit = permit
            return result
        except SliceMemberLost:
            self._latch_member_loss()
            raise

    def serve_follower(
        self,
        accelerator_step: Callable[[int], Any],
    ) -> str:
        """Execute fixed callbacks until process zero publishes termination.

        The caller must let ``SliceMemberLost`` terminate the worker process;
        swallowing it would keep a failed JAX incarnation artificially alive.
        """

        if self.is_leader:
            raise LegacySliceRuntimeError(
                "process zero cannot enter the follower loop"
            )
        if not callable(accelerator_step):
            raise TypeError("accelerator_step must be callable")
        try:
            self._require_open()
            sequence = 1
            while True:
                command = self._wait_command(sequence)
                if command.kind == "TERMINATE":
                    self._barrier(sequence, "terminated")
                    assert command.outcome is not None
                    self._terminated = True
                    return command.outcome
                if command.permit != sequence:
                    raise SliceMemberLost(
                        "follower received a non-monotonic permit"
                    )
                self._barrier(sequence, "ready")
                result = accelerator_step(sequence)
                self._block_until_ready(result)
                self._barrier(sequence, "completed")
                sequence += 1
        except SliceMemberLost:
            self._latch_member_loss()
            raise
        except BaseException as error:
            self._latch_member_loss()
            raise SliceMemberLost(
                "fixed follower accelerator callback failed"
            ) from error

    def terminate(
        self,
        *,
        outcome: str,
        incident_sha256: str | None,
    ) -> None:
        """Best-effort terminal fanout after success or incident authority."""

        del incident_sha256
        if outcome not in _OUTCOMES:
            raise ValueError("whole-slice outcome is invalid")
        with self._lock:
            if self._terminated:
                return
            self._terminated = True
        self._monitor_stop.set()
        monitor = self._monitor
        if monitor is not None and monitor is not threading.current_thread():
            monitor.join(timeout=max(1.0, self._poll_seconds * 4))
        if not self.is_leader or self._member_loss.is_set():
            return
        sequence = self._last_published_sequence + 1
        command = LegacySliceCommandV1(
            slice_binding_sha256=self._binding.digest,
            sequence=sequence,
            kind="TERMINATE",
            permit=None,
            outcome=outcome,
        )
        try:
            self._publish_command(command)
            self._last_published_sequence = sequence
            self._barrier(sequence, "terminated")
        except SliceMemberLost:
            # No step can reopen after this point. The one-shot service and JAX
            # coordinator shutdown end any follower that missed this command.
            self._member_loss.set()


__all__ = [
    "LEGACY_DEVICE_COUNT",
    "LEGACY_LOCAL_DEVICE_COUNT",
    "LEGACY_PROCESS_COUNT",
    "JAXCoordinationClientV1",
    "LegacyCoordinationClient",
    "LegacyCoordinationError",
    "LegacySliceBindingV1",
    "LegacySliceCommandV1",
    "LegacySliceRuntimeError",
    "LegacyV6e64WholeSliceRuntimeV1",
    "jax_block_until_ready_pytree",
    "jax_device_inventory_sha256",
]
