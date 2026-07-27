"""Deterministic source-only harvesting for Experiment-3 routed inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
import math
import time
from typing import Any

import numpy as np

from moe_sae.artifacts import ArtifactDataset, load_manifest
from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    MAX_SHARD_TOKENS,
    float32_to_bf16_bits,
    validate_contribution_manifest,
)
from moe_sae.contribution_harvest import (
    EXCLUDED_TOKEN_IDS,
    _validate_rebatched_source_keys,
)
from moe_sae.contribution_transcoder_artifacts import (
    RoutedInputBatch,
    validate_routed_input_batch,
)
from moe_sae.contribution_transcoder_storage import RoutedInputShardWriter
from moe_sae.harvest import (
    _block_until_ready,
    _process_local_array,
    _validate_host_grain,
    extract_grain_batch,
    make_sequence_ids,
)
from moe_sae.runtime import BoundedArtifactSink, scientific_token_mask


class RoutedInputHarvestError(RuntimeError):
    """The source-only capture diverged from its immutable parent."""


@dataclass(frozen=True, slots=True)
class RoutedInputHarvestResult:
    schema_version: str
    target_manifest_sha256: str
    source_v1_manifest_sha256: str
    split: str
    process_count: int
    process_index: int
    host_id: str
    host_tokens: int
    new_tokens: int
    resumed_tokens: int
    batches_scanned: int
    batches_executed: int
    wall_seconds: float
    device_seconds: float
    input_wait_seconds: float
    max_writer_backlog_batches: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["persisted_tokens_per_second"] = (
            self.new_tokens / self.wall_seconds
        )
        value["device_output_tokens_per_second"] = (
            0.0
            if self.device_seconds <= 0
            else self.new_tokens / self.device_seconds
        )
        return value


def routed_input_capture_to_wire(
    routed_input: Any,
    sequence_id: Any,
    position: Any,
    *,
    max_tokens: int = MAX_SHARD_TOKENS,
) -> RoutedInputBatch:
    """Convert device output to exact BF16/int64/uint16 overlay wires."""

    routed = np.asarray(routed_input)
    if routed.dtype == np.uint16:
        bits = np.ascontiguousarray(routed)
    elif routed.dtype.itemsize == 2 and routed.dtype.name == "bfloat16":
        bits = np.ascontiguousarray(routed).view(np.uint16)
    elif np.issubdtype(routed.dtype, np.floating):
        bits = float32_to_bf16_bits(routed.astype(np.float32))
    else:
        raise RoutedInputHarvestError(
            "routed_input must be floating-point or exact BF16 bits"
        )
    try:
        return validate_routed_input_batch(
            RoutedInputBatch(
                routed_input=bits,
                sequence_id=np.ascontiguousarray(
                    np.asarray(sequence_id).astype(np.int64)
                ),
                position=np.ascontiguousarray(
                    np.asarray(position).astype(np.uint16)
                ),
            ),
            max_tokens=max_tokens,
        )
    except (TypeError, ValueError) as error:
        raise RoutedInputHarvestError(
            "adapter output does not satisfy the routed-input wire schema"
        ) from error


class _RangeAccumulator:
    def __init__(self, target_entry: Mapping[str, Any]) -> None:
        self.target_entry = target_entry
        self.rows = 0
        tokens = int(target_entry["tokens"])
        self.value = RoutedInputBatch(
            routed_input=np.empty((tokens, HIDDEN_SIZE), dtype=np.uint16),
            sequence_id=np.empty((tokens,), dtype=np.int64),
            position=np.empty((tokens,), dtype=np.uint16),
        )

    @property
    def remaining(self) -> int:
        return int(self.target_entry["tokens"]) - self.rows

    @property
    def full(self) -> bool:
        return self.remaining == 0

    def append(self, value: RoutedInputBatch, offset: int) -> int:
        count = min(self.remaining, value.tokens - offset)
        if count <= 0:
            raise RoutedInputHarvestError("invalid routed-input append")
        for name in RoutedInputBatch.__dataclass_fields__:
            destination = np.asarray(getattr(self.value, name))
            source = np.asarray(getattr(value, name))
            destination[self.rows : self.rows + count] = source[
                offset : offset + count
            ]
        self.rows += count
        return count

    def take(self) -> RoutedInputBatch:
        if not self.full:
            raise RoutedInputHarvestError("cannot persist an incomplete target range")
        return validate_routed_input_batch(self.value)


def _process_target_entries(
    target: Mapping[str, Any],
    *,
    process_count: int,
    process_index: int,
) -> tuple[str, tuple[Mapping[str, Any], ...]]:
    by_host: dict[str, list[Mapping[str, Any]]] = {}
    for entry in target["shards"]:
        by_host.setdefault(str(entry["host_id"]), []).append(entry)
    if len(by_host) != process_count or not 0 <= process_index < process_count:
        raise RoutedInputHarvestError(
            "target host inventory differs from the JAX process inventory"
        )
    hosts = sorted(
        by_host,
        key=lambda host: (
            min(int(entry["source_v1_range_start"]) for entry in by_host[host]),
            host,
        ),
    )
    host_id = hosts[process_index]
    entries = tuple(
        sorted(
            by_host[host_id],
            key=lambda entry: (entry["range_start"], entry["range_id"]),
        )
    )
    if not entries:
        raise RoutedInputHarvestError("target assigned no ranges to this process")
    return host_id, entries


def _expected_keys(
    source: ArtifactDataset,
    entries: tuple[Mapping[str, Any], ...],
) -> tuple[np.ndarray, np.ndarray]:
    sequences: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for entry in entries:
        pieces = tuple(
            source.iter_token_keys_range(
                int(entry["source_v1_range_start"]),
                int(entry["source_v1_range_end"]),
            )
        )
        if not pieces:
            raise RoutedInputHarvestError("parent source range yielded no token keys")
        sequence = np.concatenate([piece[0] for piece in pieces])
        position = np.concatenate([piece[1] for piece in pieces])
        if (
            sequence.shape != (entry["tokens"],)
            or position.shape != sequence.shape
        ):
            raise RoutedInputHarvestError(
                "parent source range yielded the wrong token count"
            )
        sequences.append(sequence)
        positions.append(position)
    return np.concatenate(sequences), np.concatenate(positions)


def harvest_routed_input_host(
    batches: Iterable[Mapping[str, Any]],
    adapter_step: Callable[[Any, Any, Any, Any], Any],
    *,
    writer: RoutedInputShardWriter,
    source_v1_dataset: ArtifactDataset,
    target_manifest: Mapping[str, Any],
    process_count: int,
    process_index: int,
    sequence_length: int = 1_024,
    max_pending_writes: int = 1,
    maximum_runtime_seconds: float = 600.0,
    batch_collective: Callable[[bool, bool], tuple[bool, bool]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> RoutedInputHarvestResult:
    """Replay one source lane, capture exact inputs, and finalize its host."""

    target = validate_contribution_manifest(target_manifest)
    if type(source_v1_dataset) is not ArtifactDataset:
        raise RoutedInputHarvestError(
            "harvest requires an exact authoritative ArtifactDataset"
        )
    source = load_manifest(source_v1_dataset.root)
    source_identity = target["source_v1_manifest"]
    if (
        source["manifest_sha256"] != source_identity["manifest_sha256"]
        or source["metadata"]["split"] != target["metadata"]["split"]
    ):
        raise RoutedInputHarvestError(
            "source-v1 and contribution target identities differ"
        )
    host_id, entries = _process_target_entries(
        target, process_count=process_count, process_index=process_index
    )
    if writer.host_id != host_id:
        raise RoutedInputHarvestError("overlay writer owns the wrong target host lane")
    if writer.metadata["target_manifest_sha256"] != target["manifest_sha256"]:
        raise RoutedInputHarvestError("overlay writer binds a different target")
    if (
        sequence_length != writer.metadata["sequence_length"]
        or sequence_length != source["metadata"]["sequence_length"]
    ):
        raise RoutedInputHarvestError("source/overlay sequence lengths differ")

    expected_sequence, expected_position = _expected_keys(
        source_v1_dataset, entries
    )
    host_tokens = int(expected_sequence.size)
    committed = tuple(writer.has_target_range(entry) for entry in entries)
    seen_missing = False
    for present in committed:
        if not present:
            seen_missing = True
        elif seen_missing:
            raise RoutedInputHarvestError(
                "durable overlay ranges are not a contiguous host prefix"
            )
    completed = sum(committed)
    resumed_tokens = sum(int(entry["tokens"]) for entry in entries[:completed])
    entry_index = completed
    accumulator = (
        None if entry_index == len(entries) else _RangeAccumulator(entries[entry_index])
    )

    local_cursor = 0
    new_tokens = 0
    batches_scanned = 0
    batches_executed = 0
    device_seconds = 0.0
    input_wait_seconds = 0.0
    static_shape: tuple[int, int] | None = None
    prior_generated_sequence: int | None = None
    prior_expected_sequence: int | None = None
    started = clock()
    iterator = iter(batches)

    def persist(
        target_entry: Mapping[str, Any], value: RoutedInputBatch
    ) -> dict[str, Any]:
        return writer.write_range(target_entry, value)

    with BoundedArtifactSink(persist, max_pending=max_pending_writes) as sink:
        for batch_index in range(2**32):
            if clock() - started >= maximum_runtime_seconds:
                raise RoutedInputHarvestError(
                    "source harvest exceeded its 600-second bounded authorization"
                )
            local_complete = local_cursor >= host_tokens
            if local_complete and batch_collective is None:
                break
            wait_started = clock()
            try:
                raw = next(iterator)
            except StopIteration as error:
                raise RoutedInputHarvestError(
                    f"input exhausted at {local_cursor}/{host_tokens} host tokens"
                ) from error
            finally:
                input_wait_seconds += clock() - wait_started
            grain = extract_grain_batch(
                raw, expected_sequence_length=sequence_length
            )
            if static_shape is None:
                static_shape = grain.shape
            elif grain.shape != static_shape:
                raise RoutedInputHarvestError("static Grain shape changed")
            host_tokens_raw = _process_local_array(
                grain.token_ids,
                "transcoder inputs",
                process_count=process_count,
                process_index=process_index,
            )
            host_positions = _process_local_array(
                grain.positions,
                "transcoder positions",
                process_count=process_count,
                process_index=process_index,
            )
            host_segments = _process_local_array(
                grain.segment_ids,
                "transcoder segments",
                process_count=process_count,
                process_index=process_index,
            )
            _validate_host_grain(
                host_tokens_raw,
                host_positions,
                host_segments,
                sequence_length,
            )
            mask = scientific_token_mask(
                host_tokens_raw,
                host_segments,
                excluded_token_ids=EXCLUDED_TOKEN_IDS,
            )
            valid_indices = np.flatnonzero(mask.reshape(-1))
            desired = min(int(valid_indices.size), host_tokens - local_cursor)
            generated_ids = make_sequence_ids(
                host_segments,
                process_index=process_index,
                batch_index=batch_index,
            ).reshape(-1)
            generated_sequence = generated_ids[valid_indices[:desired]]
            generated_position = host_positions.reshape(-1)[valid_indices[:desired]]
            expected_slice = slice(local_cursor, local_cursor + desired)
            expected_batch_sequence = expected_sequence[expected_slice]
            expected_batch_position = expected_position[expected_slice]
            _validate_rebatched_source_keys(
                generated_sequence,
                generated_position,
                expected_batch_sequence,
                expected_batch_position,
                prior_generated_sequence=prior_generated_sequence,
                prior_expected_sequence=prior_expected_sequence,
            )
            if desired:
                prior_generated_sequence = int(generated_sequence[-1])
                prior_expected_sequence = int(expected_batch_sequence[-1])

            resumed_in_batch = min(
                desired, max(resumed_tokens - local_cursor, 0)
            )
            needs_execution = resumed_in_batch < desired
            local_complete_after = local_cursor + desired >= host_tokens
            if batch_collective is None:
                all_complete = local_complete_after
                any_needs_execution = needs_execution
            else:
                all_complete, any_needs_execution = batch_collective(
                    local_complete_after, needs_execution
                )
            output = None
            if any_needs_execution:
                device_started = clock()
                output = adapter_step(
                    grain.token_ids,
                    grain.positions,
                    grain.segment_ids,
                    grain.segment_ids,
                )
                _block_until_ready(output)
                device_seconds += clock() - device_started
                batches_executed += 1
            if needs_execution:
                if output is None or not hasattr(output, "routed_input"):
                    raise RoutedInputHarvestError(
                        "source adapter returned no routed_input"
                    )
                local_output = _process_local_array(
                    output.routed_input,
                    "routed-input output",
                    process_count=process_count,
                    process_index=process_index,
                    replicated_partition_unit=sequence_length,
                )
                expected_rows = int(host_tokens_raw.size)
                if local_output.shape != (expected_rows, HIDDEN_SIZE):
                    raise RoutedInputHarvestError(
                        "process-local routed input has the wrong static shape"
                    )
                selected = valid_indices[resumed_in_batch:desired]
                compacted = routed_input_capture_to_wire(
                    local_output[selected],
                    expected_batch_sequence[resumed_in_batch:desired],
                    expected_batch_position[resumed_in_batch:desired],
                )
                appended = 0
                while appended < compacted.tokens:
                    if accumulator is None:
                        raise RoutedInputHarvestError(
                            "captured more tokens than the target inventory"
                        )
                    appended += accumulator.append(compacted, appended)
                    if accumulator.full:
                        sink.submit(accumulator.target_entry, accumulator.take())
                        new_tokens += int(accumulator.target_entry["tokens"])
                        entry_index += 1
                        accumulator = (
                            None
                            if entry_index == len(entries)
                            else _RangeAccumulator(entries[entry_index])
                        )
            local_cursor += desired
            batches_scanned += 1
            if all_complete:
                break
        if local_cursor != host_tokens or accumulator is not None:
            raise RoutedInputHarvestError(
                "harvest ended before the exact target inventory was complete"
            )
    writer.finalize_host()
    wall_seconds = clock() - started
    if not math.isfinite(wall_seconds) or not 0 < wall_seconds <= maximum_runtime_seconds:
        raise RoutedInputHarvestError("source harvest exceeded its bounded runtime")
    return RoutedInputHarvestResult(
        schema_version="gemma4-routed-input-harvest-host-v1",
        target_manifest_sha256=target["manifest_sha256"],
        source_v1_manifest_sha256=source["manifest_sha256"],
        split=target["metadata"]["split"],
        process_count=process_count,
        process_index=process_index,
        host_id=host_id,
        host_tokens=host_tokens,
        new_tokens=new_tokens,
        resumed_tokens=resumed_tokens,
        batches_scanned=batches_scanned,
        batches_executed=batches_executed,
        wall_seconds=wall_seconds,
        device_seconds=device_seconds,
        input_wait_seconds=input_wait_seconds,
        max_writer_backlog_batches=sink.max_observed_backlog,
    )


__all__ = [
    "RoutedInputHarvestError",
    "RoutedInputHarvestResult",
    "harvest_routed_input_host",
    "routed_input_capture_to_wire",
]
