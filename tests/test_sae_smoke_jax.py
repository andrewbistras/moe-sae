from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from moe_sae.sae_smoke_collective import SAESmokeCollectiveError
from moe_sae.sae_smoke_jax import _JAXHooksRuntime, _payload, _row


class FakeJAX:
    @staticmethod
    def process_index() -> int:
        return 0

    @staticmethod
    def process_count() -> int:
        return 1


class FakeMultihost:
    @staticmethod
    def process_allgather(value: np.ndarray, *, tiled: bool) -> np.ndarray:
        assert tiled is False
        return np.asarray(value)[None, :]

    @staticmethod
    def broadcast_one_to_all(
        value: np.ndarray, *, is_source: bool
    ) -> np.ndarray:
        assert is_source is True
        return np.asarray(value)


def runtime() -> _JAXHooksRuntime:
    return _JAXHooksRuntime(FakeJAX(), FakeMultihost())


def test_row_round_trip_and_bounds() -> None:
    encoded = b'{"value":1}'
    assert _payload(_row(encoded, 32, label="test"), 32, label="test") == encoded
    with pytest.raises(SAESmokeCollectiveError, match="exceeds"):
        _row(b"x" * 33, 32, label="test")
    invalid = _row(b"ok", 8, label="test")
    invalid[:4] = np.frombuffer((9).to_bytes(4, "big"), dtype=np.uint8)
    with pytest.raises(SAESmokeCollectiveError, match="length"):
        _payload(invalid, 8, label="test")


def test_single_process_broadcast_and_decision_are_exact() -> None:
    hooks = runtime()
    value = hooks.broadcast_mapping("bootstrap", 0, {"b": 2, "a": 1})
    assert value == {"a": 1, "b": 2}
    reports = hooks.decision_barrier("stage", {"passed": True})
    assert reports == ({"passed": True},)


def test_broadcast_rejects_non_source_value_collectively() -> None:
    hooks = runtime()
    with pytest.raises(SAESmokeCollectiveError, match="source"):
        hooks.broadcast_mapping("bootstrap", 1, None)


def test_decision_rejects_non_json_value_collectively() -> None:
    hooks = runtime()
    with pytest.raises(SAESmokeCollectiveError, match="strict JSON"):
        hooks.decision_barrier("stage", {"bad": float("nan")})


def test_module_has_no_eager_jax_import() -> None:
    source = Path("src/moe_sae/sae_smoke_jax.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    eager = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name == "jax" or alias.name.startswith("jax.") for alias in node.names)
            if isinstance(node, ast.Import)
            else node.module == "jax" or str(node.module).startswith("jax.")
        )
    ]
    assert eager == []


def test_decision_envelope_is_canonical_json() -> None:
    hooks = runtime()
    observed = hooks.decision_barrier("canonical", {"z": 1, "a": [2, 3]})
    assert json.dumps(observed[0], sort_keys=True) == '{"a": [2, 3], "z": 1}'
