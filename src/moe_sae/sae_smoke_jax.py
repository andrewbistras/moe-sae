"""JAX multi-host hooks for the full-shape SAE smoke transaction.

The transaction protocol stays JAX-free; this small boundary adapts its two
collective primitives to ``jax.experimental.multihost_utils``.  Every payload
is strict canonical JSON with a fixed bound.  Local serialization failures are
first converted into a fixed-size all-process error decision, so process zero
cannot fail before peers enter the corresponding broadcast.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

import numpy as np

from moe_sae.sae_smoke_collective import (
    SAESmokeCollectiveError,
    SAESmokeCollectiveHooks,
)


_ERROR_BYTES = 4_096
_BROADCAST_BYTES = 8 * 1024**2
_DECISION_BYTES = 128 * 1024


def _canonical(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        if json.loads(encoded) != value:
            raise ValueError("canonical JSON round trip changed the value")
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise SAESmokeCollectiveError(
            "JAX smoke collective value must be strict JSON"
        ) from error
    return encoded


def _row(payload: bytes, capacity: int, *, label: str) -> np.ndarray:
    if len(payload) > capacity:
        raise SAESmokeCollectiveError(
            f"{label} exceeds its {capacity}-byte collective bound"
        )
    row = np.zeros((4 + capacity,), dtype=np.uint8)
    row[:4] = np.frombuffer(len(payload).to_bytes(4, "big"), dtype=np.uint8)
    if payload:
        row[4 : 4 + len(payload)] = np.frombuffer(payload, dtype=np.uint8)
    return row


def _payload(row: np.ndarray, capacity: int, *, label: str) -> bytes:
    value = np.asarray(row, dtype=np.uint8).reshape(-1)
    if value.shape != (4 + capacity,):
        raise SAESmokeCollectiveError(f"{label} collective row has the wrong shape")
    length = int.from_bytes(value[:4].tobytes(), "big")
    if not 0 <= length <= capacity:
        raise SAESmokeCollectiveError(f"{label} collective row length is invalid")
    return value[4 : 4 + length].tobytes()


class _JAXHooksRuntime:
    def __init__(self, jax_module: Any, multihost_utils: Any) -> None:
        self.jax = jax_module
        self.multihost_utils = multihost_utils
        self.process_index = int(jax_module.process_index())
        self.process_count = int(jax_module.process_count())
        if self.process_count <= 0 or not (
            0 <= self.process_index < self.process_count
        ):
            raise SAESmokeCollectiveError("JAX returned an invalid process topology")

    def _gather_rows(self, row: np.ndarray, *, label: str) -> np.ndarray:
        gathered = np.asarray(
            self.multihost_utils.process_allgather(row, tiled=False),
            dtype=np.uint8,
        )
        try:
            return gathered.reshape(self.process_count, row.shape[0])
        except ValueError as error:
            raise SAESmokeCollectiveError(
                f"{label} all-gather returned an invalid shape"
            ) from error

    def _error_gate(self, local_error: BaseException | None, *, label: str) -> None:
        message = (
            b""
            if local_error is None
            else (
                f"{type(local_error).__name__}: {local_error}"
                .encode("utf-8", errors="replace")[:_ERROR_BYTES]
            )
        )
        rows = self._gather_rows(
            _row(message, _ERROR_BYTES, label=f"{label} error"),
            label=f"{label} error",
        )
        failures: list[str] = []
        for index, row in enumerate(rows):
            payload = _payload(row, _ERROR_BYTES, label=f"{label} error")
            if payload:
                failures.append(
                    f"process {index}: "
                    + payload.decode("utf-8", errors="replace")
                )
        if failures:
            raise SAESmokeCollectiveError(
                f"{label} failed collectively: " + "; ".join(failures)
            )

    def broadcast_mapping(
        self,
        label: str,
        source_index: int,
        value: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        encoded: bytes | None = None
        local_error: BaseException | None = None
        try:
            if source_index != 0:
                raise ValueError("JAX smoke mapping source must be process zero")
            if self.process_index == source_index:
                if not isinstance(value, Mapping):
                    raise TypeError("source process supplied no mapping")
                encoded = _canonical(
                    {
                        "label": label,
                        "source_index": source_index,
                        "value": dict(value),
                    }
                )
                if len(encoded) > _BROADCAST_BYTES:
                    raise ValueError("mapping exceeds the broadcast bound")
            elif value is not None:
                raise ValueError("non-source process supplied a broadcast value")
        except BaseException as error:
            local_error = error
        self._error_gate(local_error, label=f"{label} broadcast preparation")

        local_length = (
            len(encoded)
            if self.process_index == source_index and encoded is not None
            else 0
        )
        shared_length = np.asarray(
            self.multihost_utils.broadcast_one_to_all(
                np.asarray([local_length], dtype=np.int32),
                is_source=self.process_index == source_index,
            ),
            dtype=np.int32,
        )
        if shared_length.shape != (1,):
            raise SAESmokeCollectiveError(
                f"{label} broadcast length has the wrong shape"
            )
        length = int(shared_length[0])
        if not 0 <= length <= _BROADCAST_BYTES:
            raise SAESmokeCollectiveError(f"{label} broadcast length is invalid")
        local = (
            np.frombuffer(encoded, dtype=np.uint8).copy()
            if self.process_index == source_index and encoded is not None
            else np.zeros((length,), dtype=np.uint8)
        )
        shared = np.asarray(
            self.multihost_utils.broadcast_one_to_all(
                local, is_source=self.process_index == source_index
            ),
            dtype=np.uint8,
        )
        if shared.shape != (length,):
            raise SAESmokeCollectiveError(
                f"{label} broadcast payload has the wrong shape"
            )
        try:
            envelope = json.loads(shared.tobytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SAESmokeCollectiveError(
                f"{label} broadcast payload is invalid JSON"
            ) from error
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"label", "source_index", "value"}
            or envelope.get("label") != label
            or envelope.get("source_index") != source_index
            or not isinstance(envelope.get("value"), dict)
            or _canonical(envelope) != shared.tobytes()
        ):
            raise SAESmokeCollectiveError(
                f"{label} broadcast envelope failed validation"
            )
        return dict(envelope["value"])

    def decision_barrier(
        self, label: str, value: Mapping[str, Any]
    ) -> Sequence[Mapping[str, Any]]:
        encoded: bytes | None = None
        local_error: BaseException | None = None
        try:
            if not isinstance(value, Mapping):
                raise TypeError("decision value must be a mapping")
            encoded = _canonical(
                {
                    "label": label,
                    "process_index": self.process_index,
                    "value": dict(value),
                }
            )
            if len(encoded) > _DECISION_BYTES:
                raise ValueError("decision value exceeds the all-gather bound")
        except BaseException as error:
            local_error = error
        self._error_gate(local_error, label=f"{label} decision preparation")
        assert encoded is not None
        rows = self._gather_rows(
            _row(encoded, _DECISION_BYTES, label=f"{label} decision"),
            label=f"{label} decision",
        )
        by_process: dict[int, Mapping[str, Any]] = {}
        for row in rows:
            raw = _payload(row, _DECISION_BYTES, label=f"{label} decision")
            try:
                envelope = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SAESmokeCollectiveError(
                    f"{label} decision contains invalid JSON"
                ) from error
            if (
                not isinstance(envelope, dict)
                or set(envelope) != {"label", "process_index", "value"}
                or envelope.get("label") != label
                or not isinstance(envelope.get("process_index"), int)
                or isinstance(envelope.get("process_index"), bool)
                or not isinstance(envelope.get("value"), dict)
                or _canonical(envelope) != raw
            ):
                raise SAESmokeCollectiveError(
                    f"{label} decision envelope failed validation"
                )
            index = int(envelope["process_index"])
            if not 0 <= index < self.process_count or index in by_process:
                raise SAESmokeCollectiveError(
                    f"{label} decision process membership is invalid"
                )
            by_process[index] = dict(envelope["value"])
        if tuple(sorted(by_process)) != tuple(range(self.process_count)):
            raise SAESmokeCollectiveError(
                f"{label} decision omitted a JAX process"
            )
        return tuple(by_process[index] for index in range(self.process_count))


def jax_collective_hooks() -> SAESmokeCollectiveHooks:
    """Build production hooks after JAX distributed initialization."""

    import jax
    from jax.experimental import multihost_utils

    runtime = _JAXHooksRuntime(jax, multihost_utils)
    return SAESmokeCollectiveHooks(
        process_index=runtime.process_index,
        process_count=runtime.process_count,
        coordinator_index=0,
        broadcast_mapping=runtime.broadcast_mapping,
        decision_barrier=runtime.decision_barrier,
    )


__all__ = ["jax_collective_hooks"]
