from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import itertools
import threading

import pytest

from moe_sae.r1.control_store import (
    CASConflict,
    ControlStoreError,
    ImmutableConflict,
    MemoryJSONCASStore,
)
from moe_sae.r1.protocol import ProtocolError


def _state(value: int) -> dict[str, object]:
    return {"schema": "test.controller-state.v1", "value": value}


def test_create_read_and_cas_use_injected_generations() -> None:
    generations = iter((11, 19, 42))
    store = MemoryJSONCASStore(generation_factory=generations.__next__)

    created = store.create("memory://campaign/state", _state(1))
    updated = store.compare_and_swap(
        "memory://campaign/state", created.generation, _state(2)
    )

    assert created.generation == 11
    assert updated.generation == 19
    assert store.read("memory://campaign/state") == updated
    assert updated.decoded() == _state(2)
    assert updated.value == b'{"schema":"test.controller-state.v1","value":2}'


def test_create_is_absent_only_and_reports_actual_generation() -> None:
    store = MemoryJSONCASStore()
    existing = store.create("state", _state(1))
    with pytest.raises(CASConflict) as raised:
        store.create("state", _state(2))
    assert raised.value.expected_generation is None
    assert raised.value.actual_generation == existing.generation


def test_cas_rejects_absent_stale_and_boolean_generations() -> None:
    store = MemoryJSONCASStore()
    created = store.create("state", _state(1))

    with pytest.raises(CASConflict) as absent:
        store.compare_and_swap("missing", created.generation, _state(2))
    assert absent.value.actual_generation is None

    updated = store.compare_and_swap("state", created.generation, _state(2))
    with pytest.raises(CASConflict) as stale:
        store.compare_and_swap("state", created.generation, _state(3))
    assert stale.value.actual_generation == updated.generation

    with pytest.raises(ValueError, match="positive integer"):
        store.compare_and_swap("state", True, _state(4))


def test_two_way_cas_race_has_exactly_one_winner() -> None:
    store = MemoryJSONCASStore()
    initial = store.create("state", _state(0))
    barrier = threading.Barrier(2)

    def contender(value: int) -> tuple[str, int]:
        barrier.wait()
        try:
            result = store.compare_and_swap("state", initial.generation, _state(value))
            return "won", result.generation
        except CASConflict as error:
            assert error.expected_generation == initial.generation
            assert error.actual_generation is not None
            return "lost", error.actual_generation

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(contender, (1, 2)))

    assert sorted(outcome for outcome, _ in results) == ["lost", "won"]
    assert store.read("state").decoded()["value"] in {1, 2}  # type: ignore[union-attr]


def test_immutable_put_is_idempotent_but_never_replaceable() -> None:
    store = MemoryJSONCASStore()
    first = store.put_immutable("event/1", _state(1))
    retry = store.put_immutable(
        "event/1", b'{ "value": 1, "schema": "test.controller-state.v1" }'
    )

    assert retry == first
    with pytest.raises(ImmutableConflict, match="different content"):
        store.put_immutable("event/1", _state(2))
    with pytest.raises(ImmutableConflict, match="cannot be replaced"):
        store.compare_and_swap("event/1", first.generation, _state(2))


def test_mutable_name_cannot_be_reclassified_as_immutable() -> None:
    store = MemoryJSONCASStore()
    store.create("state", _state(1))
    with pytest.raises(ImmutableConflict, match="mutable"):
        store.put_immutable("state", _state(1))


def test_concurrent_identical_immutable_publication_is_one_generation() -> None:
    store = MemoryJSONCASStore()
    barrier = threading.Barrier(8)

    def publish(_: int) -> int:
        barrier.wait()
        return store.put_immutable("event/1", _state(1)).generation

    with ThreadPoolExecutor(max_workers=8) as executor:
        generations = list(executor.map(publish, range(8)))

    assert len(set(generations)) == 1


def test_values_require_versioned_object_schema_and_finite_json() -> None:
    store = MemoryJSONCASStore()
    with pytest.raises(ProtocolError, match="must be an object"):
        store.create("x", [1, 2, 3])
    with pytest.raises(ProtocolError, match="versioned schema"):
        store.create("x", {"value": 1})
    with pytest.raises(ProtocolError, match="versioned schema"):
        store.create("x", {"schema": "unversioned", "value": 1})
    with pytest.raises(ProtocolError, match="non-finite"):
        store.create("x", '{"schema":"test.v1","value":Infinity}')


def test_failed_precondition_does_not_consume_generation() -> None:
    generations = iter((5, 9))
    store = MemoryJSONCASStore(generation_factory=generations.__next__)
    first = store.create("state", _state(1))
    with pytest.raises(CASConflict):
        store.create("state", _state(2))
    second = store.create("other", _state(3))
    assert (first.generation, second.generation) == (5, 9)


def test_generation_factory_must_be_strictly_increasing_integer() -> None:
    generations = iter((5, 5))
    store = MemoryJSONCASStore(generation_factory=generations.__next__)
    store.create("one", _state(1))
    with pytest.raises(ControlStoreError, match="strictly increasing"):
        store.create("two", _state(2))
    assert store.read("two") is None

    bool_store = MemoryJSONCASStore(generation_factory=itertools.repeat(True).__next__)
    with pytest.raises(ControlStoreError, match="strictly increasing"):
        bool_store.create("one", _state(1))


def test_versioned_json_is_frozen() -> None:
    version = MemoryJSONCASStore().create("state", _state(1))
    with pytest.raises(FrozenInstanceError):
        version.generation = 9  # type: ignore[misc]
