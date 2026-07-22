from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import threading
import uuid

import pytest

from moe_sae.r1.control_store import CASConflict, ImmutableConflict
from moe_sae.r1.gcs_control_store import GCSJSONCASStore


def _state(value: int) -> dict[str, object]:
    return {"schema": "test.real-gcs-controller-state.v1", "value": value}


def _object(prefix: str, run: str, name: str) -> str:
    prefix = prefix.rstrip("/")
    if not prefix.startswith("gs://") or "/" not in prefix[5:]:
        raise ValueError("MOE_SAE_TEST_GCS_PREFIX must include a bucket and prefix")
    return f"{prefix}/pytest-r1-control/{run}/{name}"


def _parts(uri: str) -> tuple[str, str]:
    bucket, separator, name = uri[5:].partition("/")
    if not separator or not bucket or not name:
        raise ValueError("invalid GCS URI")
    return bucket, name


@pytest.mark.skipif(
    not os.environ.get("MOE_SAE_TEST_GCS_PREFIX"),
    reason=("set MOE_SAE_TEST_GCS_PREFIX to opt into a real generation/CAS GCS test"),
)
def test_real_gcs_generation_race_and_immutable_round_trip() -> None:
    """Exercise real preconditions without ever listing the test prefix."""

    from google.cloud import storage

    prefix = os.environ["MOE_SAE_TEST_GCS_PREFIX"]
    run = uuid.uuid4().hex
    head_uri = _object(prefix, run, "head.json")
    event_uri = _object(prefix, run, "event.json")
    client = storage.Client()
    store = GCSJSONCASStore(client)
    known_uris = (head_uri, event_uri)

    try:
        created = store.create(head_uri, _state(0))
        barrier = threading.Barrier(2)

        def contender(value: int) -> tuple[str, int]:
            barrier.wait()
            try:
                result = store.compare_and_swap(
                    head_uri, created.generation, _state(value)
                )
                return "won", result.generation
            except CASConflict as error:
                assert error.actual_generation is not None
                return "lost", error.actual_generation

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(contender, (1, 2)))

        assert sorted(outcome for outcome, _generation in results) == ["lost", "won"]
        assert len({generation for _outcome, generation in results}) == 1
        current = store.read(head_uri)
        assert current is not None
        assert current.decoded()["value"] in {1, 2}

        immutable = store.put_immutable(event_uri, _state(7))
        assert store.put_immutable(event_uri, _state(7)) == immutable
        with pytest.raises(ImmutableConflict, match="different content"):
            store.put_immutable(event_uri, _state(8))
        with pytest.raises(ImmutableConflict, match="cannot be replaced"):
            store.compare_and_swap(event_uri, immutable.generation, _state(8))
    finally:
        # The test knows every object name it created.  Cleanup also avoids a
        # listing and deletes only the exact generation observed by reload.
        for uri in known_uris:
            bucket_name, object_name = _parts(uri)
            blob = client.bucket(bucket_name).blob(object_name)
            try:
                blob.reload()
                blob.delete(if_generation_match=int(blob.generation))
            except Exception as error:
                if int(getattr(error, "code", 0) or 0) != 404:
                    raise
