"""CPU-testable Experiment-2 contribution harvesting orchestration.

Model execution is injected as one static-shape callable.  This module owns
the scientific-token stream, exact v1 alignment, logical/source range plan,
wire conversion, bounded persistence, and deterministic resume.  It neither
imports JAX nor constructs MaxText at module import time.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from moe_sae.artifacts import ArtifactDataset, load_manifest
from moe_sae.contribution_artifacts import (
    HIDDEN_SIZE,
    MAX_SHARD_TOKENS,
    TOP_K_EXPERTS,
    ContributionBatch,
    float32_to_bf16_bits,
    validate_contribution_batch,
)
from moe_sae.contribution_ranges import (
    ContributionRangeAssignment,
    ContributionRangePlan,
)
from moe_sae.contribution_recomposition import (
    decode_recomposition_sidecar,
    maxtext_rmsnorm_bf16,
)
from moe_sae.contribution_storage import ContributionShardWriter
from moe_sae.harvest import (
    _block_until_ready,
    _process_local_array,
    _validate_host_grain,
    extract_grain_batch,
    make_sequence_ids,
)
from moe_sae.runtime import BoundedArtifactSink, scientific_token_mask

EXCLUDED_TOKEN_IDS = (0, 1, 2)


class ContributionHarvestError(RuntimeError):
    """The deterministic Experiment-2 harvest contract was violated."""


@dataclass(frozen=True, slots=True)
class ContributionHarvestResult:
    schema_version: str
    range_plan_sha256: str
    source_v1_manifest_sha256: str
    split: str
    phase_total_tokens: int
    process_count: int
    process_index: int
    host_id: str
    host_tokens: int
    new_tokens: int
    resumed_tokens: int
    batches_scanned: int
    batches_executed: int
    executed_scientific_tokens: int
    wall_seconds: float
    device_seconds: float
    input_wait_seconds: float
    max_writer_backlog_batches: int

    @property
    def input_wait_fraction(self) -> float:
        return self.input_wait_seconds / self.wall_seconds

    @property
    def device_output_tokens_per_second(self) -> float:
        if self.device_seconds <= 0:
            return 0.0
        return self.executed_scientific_tokens / self.device_seconds

    @property
    def persisted_tokens_per_second(self) -> float:
        return self.new_tokens / self.wall_seconds

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "input_wait_fraction": self.input_wait_fraction,
                "device_output_tokens_per_second": (
                    self.device_output_tokens_per_second
                ),
                "persisted_tokens_per_second": self.persisted_tokens_per_second,
            }
        )
        return value


def _to_bf16_bits(name: str, value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.uint16:
        bits = np.ascontiguousarray(array)
    elif array.dtype.itemsize == 2 and array.dtype.name == "bfloat16":
        # NumPy classifies ml_dtypes.bfloat16 as kind ``V`` rather than a
        # floating subtype.  Its in-memory representation is already the exact
        # uint16 BF16 wire representation, so bitcast it without an expensive
        # BF16 -> FP32 -> BF16 round trip.
        bits = np.ascontiguousarray(array).view(np.uint16)
    elif np.issubdtype(array.dtype, np.floating):
        bits = float32_to_bf16_bits(np.asarray(array, dtype=np.float32))
    else:
        raise ContributionHarvestError(
            f"{name} must be floating-point or exact BF16 uint16 bits"
        )
    return bits


def contribution_capture_to_wire(
    *,
    expert_output: Any,
    expert_ids: Any,
    router_weights: Any,
    shared_output: Any,
    sequence_id: Any,
    position: Any,
    max_tokens: int = MAX_SHARD_TOKENS,
) -> ContributionBatch:
    """Convert one process-local compacted capture to the exact v2 wire."""

    ids = np.asarray(expert_ids)
    sequences = np.asarray(sequence_id)
    positions = np.asarray(position)
    if not np.issubdtype(ids.dtype, np.integer):
        raise ContributionHarvestError("expert_ids must be integer")
    if not np.issubdtype(sequences.dtype, np.signedinteger):
        raise ContributionHarvestError("sequence_id must be a signed integer")
    if not np.issubdtype(positions.dtype, np.integer):
        raise ContributionHarvestError("position must be integer")
    try:
        result = ContributionBatch(
            expert_output=_to_bf16_bits("expert_output", expert_output),
            expert_ids=np.ascontiguousarray(ids.astype(np.uint8)),
            router_weights=_to_bf16_bits("router_weights", router_weights),
            shared_output=_to_bf16_bits("shared_output", shared_output),
            sequence_id=np.ascontiguousarray(sequences.astype(np.int64)),
            position=np.ascontiguousarray(positions.astype(np.uint16)),
        )
        return validate_contribution_batch(result, max_tokens=max_tokens)
    except (ValueError, TypeError) as error:
        raise ContributionHarvestError(
            "adapter output does not satisfy the contribution-v2 wire schema"
        ) from error


def contribution_capture_to_wire_chunks(
    *,
    expert_output: Any,
    expert_ids: Any,
    router_weights: Any,
    shared_output: Any,
    sequence_id: Any,
    position: Any,
    max_tokens: int = MAX_SHARD_TOKENS,
) -> tuple[ContributionBatch, ...]:
    """Split one process-local output into independently valid wire batches."""

    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 0 < max_tokens <= MAX_SHARD_TOKENS
    ):
        raise ContributionHarvestError(
            f"max_tokens must be in [1,{MAX_SHARD_TOKENS}]"
        )
    arrays = {
        "expert_output": np.asarray(expert_output),
        "expert_ids": np.asarray(expert_ids),
        "router_weights": np.asarray(router_weights),
        "shared_output": np.asarray(shared_output),
        "sequence_id": np.asarray(sequence_id),
        "position": np.asarray(position),
    }
    if any(value.ndim == 0 for value in arrays.values()):
        raise ContributionHarvestError(
            "adapter output fields must expose a leading token axis"
        )
    token_counts = {int(value.shape[0]) for value in arrays.values()}
    if len(token_counts) != 1:
        raise ContributionHarvestError(
            "adapter output fields disagree on their leading token count"
        )
    tokens = token_counts.pop()
    if tokens <= 0:
        raise ContributionHarvestError("adapter output contains no tokens")
    return tuple(
        contribution_capture_to_wire(
            **{
                name: value[start : start + max_tokens]
                for name, value in arrays.items()
            },
            max_tokens=max_tokens,
        )
        for start in range(0, tokens, max_tokens)
    )


class _AssignmentAccumulator:
    def __init__(self, assignment: ContributionRangeAssignment) -> None:
        self.assignment = assignment
        rows = assignment.tokens
        self.rows = 0
        self.value = ContributionBatch(
            expert_output=np.empty(
                (rows, TOP_K_EXPERTS, HIDDEN_SIZE),
                dtype=np.uint16,
            ),
            expert_ids=np.empty((rows, TOP_K_EXPERTS), dtype=np.uint8),
            router_weights=np.empty((rows, TOP_K_EXPERTS), dtype=np.uint16),
            shared_output=np.empty((rows, HIDDEN_SIZE), dtype=np.uint16),
            sequence_id=np.empty((rows,), dtype=np.int64),
            position=np.empty((rows,), dtype=np.uint16),
        )

    @property
    def remaining(self) -> int:
        return self.assignment.tokens - self.rows

    @property
    def full(self) -> bool:
        return self.remaining == 0

    def append(self, value: ContributionBatch, offset: int) -> int:
        if self.full:
            raise ContributionHarvestError("cannot append to a full range")
        if not 0 <= offset < value.tokens:
            raise ContributionHarvestError("capture append offset is outside its batch")
        count = min(self.remaining, value.tokens - offset)
        for name in ContributionBatch.__dataclass_fields__:
            destination = np.asarray(getattr(self.value, name))
            source = np.asarray(getattr(value, name))
            destination[self.rows : self.rows + count] = source[offset : offset + count]
        self.rows += count
        return count

    def take(self) -> ContributionBatch:
        if not self.full:
            raise ContributionHarvestError("cannot persist an incomplete range")
        return validate_contribution_batch(self.value)


def _source_keys(
    source: ArtifactDataset,
    assignments: tuple[ContributionRangeAssignment, ...],
) -> tuple[np.ndarray, np.ndarray]:
    sequence_parts: list[np.ndarray] = []
    position_parts: list[np.ndarray] = []
    for assignment in assignments:
        pieces = tuple(
            source.iter_token_keys_range(
                assignment.source_v1_range_start,
                assignment.source_v1_range_end,
            )
        )
        if not pieces:
            raise ContributionHarvestError(
                f"source-v1 range {assignment.ordinal} yielded no token keys"
            )
        sequence = np.concatenate([piece[0] for piece in pieces])
        position = np.concatenate([piece[1] for piece in pieces])
        if sequence.shape != (assignment.tokens,) or position.shape != sequence.shape:
            raise ContributionHarvestError(
                f"source-v1 range {assignment.ordinal} has the wrong token count"
            )
        sequence_parts.append(sequence)
        position_parts.append(position)
    return np.concatenate(sequence_parts), np.concatenate(position_parts)


def _validate_rebatched_source_keys(
    generated_sequence: np.ndarray,
    generated_position: np.ndarray,
    expected_sequence: np.ndarray,
    expected_position: np.ndarray,
    *,
    prior_generated_sequence: int | None,
    prior_expected_sequence: int | None,
) -> None:
    """Prove one staged token slice is v1-identical across static rebatching.

    V1 sequence IDs encode ``process | batch | row | packed-segment``.  A
    larger static sequence batch changes only the middle ``batch | row`` bits,
    even when it consumes the exact same immutable ArrayRecord stream.  Those
    incidental bits cannot be compared numerically when Experiment 2 raises
    batch 1 to batch 8.

    The invariant identity is stronger than position alone: process lane,
    packed segment ID, token position, and the complete sequence-boundary
    partition must agree.  The original v1 IDs are then written to v2, keeping
    the scientific token inventory bit-identical to the sealed source.
    """

    generated = np.asarray(generated_sequence, dtype=np.int64)
    expected = np.asarray(expected_sequence, dtype=np.int64)
    generated_positions = np.asarray(generated_position, dtype=np.uint16)
    expected_positions = np.asarray(expected_position, dtype=np.uint16)
    if (
        generated.ndim != 1
        or expected.shape != generated.shape
        or generated_positions.shape != generated.shape
        or expected_positions.shape != generated.shape
    ):
        raise ContributionHarvestError(
            "staged/source token-key slices have incompatible shapes"
        )
    if generated.size == 0:
        return
    segment_mask = np.int64((1 << 16) - 1)
    process_mask = np.int64(((1 << 7) - 1) << 56)
    if (
        not np.array_equal(generated_positions, expected_positions)
        or not np.array_equal(
            generated & segment_mask,
            expected & segment_mask,
        )
        or not np.array_equal(
            generated & process_mask,
            expected & process_mask,
        )
    ):
        raise ContributionHarvestError(
            "staged input token keys differ from the sealed source-v1 lane"
        )
    generated_boundaries = np.empty(generated.shape, dtype=np.bool_)
    expected_boundaries = np.empty(expected.shape, dtype=np.bool_)
    generated_boundaries[0] = (
        prior_generated_sequence is None or generated[0] != prior_generated_sequence
    )
    expected_boundaries[0] = (
        prior_expected_sequence is None or expected[0] != prior_expected_sequence
    )
    generated_boundaries[1:] = generated[1:] != generated[:-1]
    expected_boundaries[1:] = expected[1:] != expected[:-1]
    if not np.array_equal(generated_boundaries, expected_boundaries):
        raise ContributionHarvestError(
            "staged input sequence boundaries differ from the sealed " "source-v1 lane"
        )


def _localize_adapter_output(
    output: Any,
    *,
    process_count: int,
    process_index: int,
    sequence_length: int,
    expected_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fields = []
    for name in (
        "expert_output",
        "expert_ids",
        "router_weights",
        "shared_output",
    ):
        if not hasattr(output, name):
            raise ContributionHarvestError(f"contribution adapter output lacks {name}")
        fields.append(
            _process_local_array(
                getattr(output, name),
                f"contribution {name}",
                process_count=process_count,
                process_index=process_index,
                replicated_partition_unit=sequence_length,
            )
        )
    expert, ids, weights, shared = fields
    if expert.shape != (expected_rows, TOP_K_EXPERTS, HIDDEN_SIZE):
        raise ContributionHarvestError(
            "process-local selected expert output has the wrong static shape"
        )
    if ids.shape != (expected_rows, TOP_K_EXPERTS):
        raise ContributionHarvestError(
            "process-local expert IDs have the wrong static shape"
        )
    if weights.shape != ids.shape:
        raise ContributionHarvestError(
            "process-local router weights have the wrong static shape"
        )
    if shared.shape != (expected_rows, HIDDEN_SIZE):
        raise ContributionHarvestError(
            "process-local shared output has the wrong static shape"
        )
    return expert, ids, weights, shared


def harvest_contribution_host(
    batches: Iterable[Mapping[str, Any]],
    adapter_step: Callable[[Any, Any, Any, Any], Any],
    *,
    writer: ContributionShardWriter,
    source_v1_dataset: ArtifactDataset,
    range_plan: ContributionRangePlan,
    phase_total_tokens: int,
    process_count: int,
    process_index: int,
    recomposition_sidecar: Mapping[str, Any],
    sequence_length: int = 1_024,
    max_pending_writes: int = 1,
    batch_collective: Callable[[bool, bool], tuple[bool, bool]] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> ContributionHarvestResult:
    """Capture one source lane and finalize its exact phase host index."""

    if type(source_v1_dataset) is not ArtifactDataset:
        raise ContributionHarvestError(
            "authoritative harvest requires an exact sealed ArtifactDataset"
        )
    source_manifest = load_manifest(source_v1_dataset.root)
    if source_manifest["manifest_sha256"] != range_plan.source_manifest_sha256:
        raise ContributionHarvestError(
            "range plan and source-v1 dataset bind different manifests"
        )
    if process_count != range_plan.process_count:
        raise ContributionHarvestError(
            "runtime process count differs from the frozen range plan"
        )
    assignments = range_plan.for_process(phase_total_tokens, process_index)
    if not assignments:
        raise ContributionHarvestError(
            "this process has no planned contribution ranges"
        )
    host_id = assignments[0].host_id
    if writer.host_id != host_id:
        raise ContributionHarvestError("writer host ID differs from its source-v1 lane")
    if (
        writer.metadata["source_v1_manifest_sha256"]
        != range_plan.source_manifest_sha256
    ):
        raise ContributionHarvestError(
            "writer metadata differs from the range-plan source manifest"
        )
    if sequence_length != int(
        writer.metadata["sequence_length"]
    ) or sequence_length != int(source_manifest["metadata"]["sequence_length"]):
        raise ContributionHarvestError(
            "runtime sequence length differs from the bound v1/v2 metadata"
        )
    expected_sequence, expected_position = _source_keys(
        source_v1_dataset,
        assignments,
    )
    host_tokens = int(expected_sequence.size)
    if host_tokens != phase_total_tokens // process_count:
        raise ContributionHarvestError(
            "range plan does not allocate the exact balanced host token quota"
        )

    range_ids = tuple(
        writer.range_id(
            assignment.range_start,
            assignment.range_end,
            expected_sequence[
                sum(item.tokens for item in assignments[:index]) : sum(
                    item.tokens for item in assignments[: index + 1]
                )
            ],
            expected_position[
                sum(item.tokens for item in assignments[:index]) : sum(
                    item.tokens for item in assignments[: index + 1]
                )
            ],
            source_v1_range_start=assignment.source_v1_range_start,
            source_v1_range_end=assignment.source_v1_range_end,
        )
        for index, assignment in enumerate(assignments)
    )
    committed = tuple(
        writer.has_range(
            range_id,
            assignment.range_start,
            assignment.range_end,
        )
        for assignment, range_id in zip(assignments, range_ids, strict=True)
    )
    seen_missing = False
    for present in committed:
        if not present:
            seen_missing = True
        elif seen_missing:
            raise ContributionHarvestError(
                "durable contribution ranges are not a contiguous host prefix"
            )
    completed_ranges = sum(committed)
    resumed_tokens = sum(
        assignment.tokens for assignment in assignments[:completed_ranges]
    )

    sidecar = decode_recomposition_sidecar(recomposition_sidecar)
    assignment_index = completed_ranges
    accumulator = (
        None
        if assignment_index == len(assignments)
        else _AssignmentAccumulator(assignments[assignment_index])
    )
    new_tokens = 0
    batches_scanned = 0
    batches_executed = 0
    executed_scientific_tokens = 0
    input_wait_seconds = 0.0
    device_seconds = 0.0
    static_shape: tuple[int, int] | None = None
    local_cursor = 0
    prior_generated_sequence: int | None = None
    prior_expected_sequence: int | None = None
    started = clock()

    def persist(
        assignment: ContributionRangeAssignment,
        range_id: str,
        value: ContributionBatch,
        source_sequence: np.ndarray,
        source_position: np.ndarray,
    ) -> dict[str, Any]:
        _, normalized_shared = maxtext_rmsnorm_bf16(
            value.shared_output,
            sidecar.shared_rmsnorm_scale_bf16_bits,
            epsilon=float(sidecar.epsilon),
        )
        return writer.write_range(
            range_id,
            assignment.range_start,
            assignment.range_end,
            value,
            source_v1_range_start=assignment.source_v1_range_start,
            source_v1_range_end=assignment.source_v1_range_end,
            normalized_shared_output=normalized_shared,
            source_sequence_id=source_sequence,
            source_position=source_position,
        )

    iterator = iter(batches)
    with BoundedArtifactSink(persist, max_pending=max_pending_writes) as sink:
        for batch_index in range(2**32):
            local_complete = local_cursor >= host_tokens
            if local_complete and batch_collective is None:
                break
            wait_started = clock()
            try:
                raw = next(iterator)
            except StopIteration as error:
                input_wait_seconds += clock() - wait_started
                raise ContributionHarvestError(
                    f"input exhausted at {local_cursor}/{host_tokens} host tokens"
                ) from error
            input_wait_seconds += clock() - wait_started
            grain = extract_grain_batch(
                raw,
                expected_sequence_length=sequence_length,
            )
            if static_shape is None:
                static_shape = grain.shape
            elif grain.shape != static_shape:
                raise ContributionHarvestError(
                    f"static Grain shape changed from {static_shape} to {grain.shape}"
                )
            host_token_ids = _process_local_array(
                grain.token_ids,
                "contribution inputs",
                process_count=process_count,
                process_index=process_index,
            )
            host_positions = _process_local_array(
                grain.positions,
                "contribution positions",
                process_count=process_count,
                process_index=process_index,
            )
            host_segments = _process_local_array(
                grain.segment_ids,
                "contribution segmentation",
                process_count=process_count,
                process_index=process_index,
            )
            _validate_host_grain(
                host_token_ids,
                host_positions,
                host_segments,
                sequence_length,
            )
            mask = scientific_token_mask(
                host_token_ids,
                host_segments,
                excluded_token_ids=EXCLUDED_TOKEN_IDS,
            )
            valid_indices = np.flatnonzero(mask.reshape(-1))
            desired = min(int(valid_indices.size), host_tokens - local_cursor)
            host_sequence_ids = make_sequence_ids(
                host_segments,
                process_index=process_index,
                batch_index=batch_index,
            ).reshape(-1)
            generated_sequence = host_sequence_ids[valid_indices[:desired]]
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
                desired,
                max(resumed_tokens - local_cursor, 0),
            )
            needs_execution = resumed_in_batch < desired
            local_complete_after = local_cursor + desired >= host_tokens
            if batch_collective is None:
                all_complete = local_complete_after
                any_needs_execution = needs_execution
            else:
                all_complete, any_needs_execution = batch_collective(
                    local_complete_after,
                    needs_execution,
                )

            output = None
            if any_needs_execution:
                device_started = clock()
                try:
                    output = adapter_step(
                        grain.token_ids,
                        grain.positions,
                        grain.segment_ids,
                        grain.segment_ids,
                    )
                    _block_until_ready(output)
                finally:
                    device_seconds += clock() - device_started
                batches_executed += 1
                executed_scientific_tokens += int(valid_indices.size)
            if needs_execution:
                if output is None:
                    raise ContributionHarvestError(
                        "collective skipped a locally required adapter execution"
                    )
                rows = int(host_token_ids.size)
                expert, ids, weights, shared = _localize_adapter_output(
                    output,
                    process_count=process_count,
                    process_index=process_index,
                    sequence_length=sequence_length,
                    expected_rows=rows,
                )
                selected_indices = valid_indices[resumed_in_batch:desired]
                new_position = host_positions.reshape(-1)[selected_indices]
                # Retain the authoritative v1 IDs.  The live IDs encode this
                # run's batch/row grouping and therefore legitimately differ
                # when a qualified larger static batch replays the same
                # immutable record stream.
                new_sequence = expected_batch_sequence[resumed_in_batch:desired]
                compacted_chunks = contribution_capture_to_wire_chunks(
                    expert_output=expert[selected_indices],
                    expert_ids=ids[selected_indices],
                    router_weights=weights[selected_indices],
                    shared_output=shared[selected_indices],
                    sequence_id=new_sequence,
                    position=new_position,
                )
                for compacted in compacted_chunks:
                    appended = 0
                    while appended < compacted.tokens:
                        if accumulator is None:
                            raise ContributionHarvestError(
                                "captured more tokens than the range plan permits"
                            )
                        appended += accumulator.append(compacted, appended)
                        if accumulator.full:
                            assignment = accumulator.assignment
                            assignment_offset = sum(
                                item.tokens
                                for item in assignments[:assignment_index]
                            )
                            source_slice = slice(
                                assignment_offset,
                                assignment_offset + assignment.tokens,
                            )
                            sink.submit(
                                assignment,
                                range_ids[assignment_index],
                                accumulator.take(),
                                expected_sequence[source_slice],
                                expected_position[source_slice],
                            )
                            new_tokens += assignment.tokens
                            assignment_index += 1
                            accumulator = (
                                None
                                if assignment_index == len(assignments)
                                else _AssignmentAccumulator(
                                    assignments[assignment_index]
                                )
                            )

            local_cursor += desired
            batches_scanned += 1
            if all_complete:
                break
        if local_cursor != host_tokens:
            raise ContributionHarvestError(
                f"harvest stopped at {local_cursor}/{host_tokens} host tokens"
            )
        if accumulator is not None:
            raise ContributionHarvestError(
                "harvest ended with an incomplete planned contribution range"
            )
    writer.finalize_host()
    wall_seconds = clock() - started
    if not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ContributionHarvestError("harvest wall time is not positive")
    return ContributionHarvestResult(
        schema_version="gemma4-contribution-harvest-host-v1",
        range_plan_sha256=range_plan.plan_sha256,
        source_v1_manifest_sha256=range_plan.source_manifest_sha256,
        split=writer.metadata["split"],
        phase_total_tokens=phase_total_tokens,
        process_count=process_count,
        process_index=process_index,
        host_id=host_id,
        host_tokens=host_tokens,
        new_tokens=new_tokens,
        resumed_tokens=resumed_tokens,
        batches_scanned=batches_scanned,
        batches_executed=batches_executed,
        executed_scientific_tokens=executed_scientific_tokens,
        wall_seconds=wall_seconds,
        device_seconds=device_seconds,
        input_wait_seconds=input_wait_seconds,
        max_writer_backlog_batches=sink.max_observed_backlog,
    )


__all__ = [
    "ContributionHarvestError",
    "ContributionHarvestResult",
    "contribution_capture_to_wire",
    "harvest_contribution_host",
]
