from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, NamedTuple

import numpy as np
import pytest

from moe_sae.sae_smoke import (
    SAEFullShape,
    SAESmokeAttemptFailed,
    SAESmokeIntegrityError,
    SAESmokeRunner,
    SmokeAttemptIdentity,
    SmokeCampaignIdentity,
)
from moe_sae.sae_smoke_tpu import (
    GCSObjectVersion,
    OrbaxPayloadInspection,
    RestoredPayload,
    SAESmokeTPUAdapter,
    StateProgress,
    TPUSmokeRuntime,
    checkpoint_inventory_sha256,
    hash_addressable_train_state,
    require_identical_host_digests,
)


def digest(character: str) -> str:
    return character * 64


@dataclass(frozen=True)
class FakeSAEConfig:
    condition: str
    hidden_dim: int = 2_816
    num_experts: int = 128
    top_routed: int = 8
    bank_width: int = 1_024
    k: int = 64
    parameter_dtype: str = "bfloat16"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    peak_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 250
    total_steps: int = 3_052

    @property
    def total_features(self) -> int:
        return self.top_routed * self.bank_width if self.condition == "B" else (
            self.num_experts * self.bank_width
        )

    @property
    def candidates_per_token(self) -> int:
        return (
            self.top_routed * self.bank_width
            if self.condition in {"A", "B"}
            else self.num_experts * self.bank_width
        )

    @property
    def parameter_count(self) -> int:
        return self.total_features * (2 * self.hidden_dim + 1)


@dataclass(frozen=True)
class FakeTrainingConfig:
    sae: FakeSAEConfig
    seed: int = 1
    total_tokens: int = 50_000_000
    effective_batch_tokens: int = 16_384
    microbatch_tokens: int = 4_096
    checkpoint_every_tokens: int = 10_000_000
    health_gate_tokens: int = 5_000_000
    live_feature_fraction: float = 0.95
    loss_ratio_threshold: float = 0.80
    a_capacity: int | None = None

    @property
    def microbatches_per_update(self) -> int:
        return self.effective_batch_tokens // self.microbatch_tokens

    @property
    def updates(self) -> int:
        return (
            self.total_tokens + self.effective_batch_tokens - 1
        ) // self.effective_batch_tokens

    @property
    def final_padding(self) -> int:
        return self.updates * self.effective_batch_tokens - self.total_tokens


class FakeArray:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str,
        *,
        scalar: int | None = None,
        addressable_shards: tuple[object, ...] = (),
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.scalar = scalar
        self.addressable_shards = addressable_shards


class Params(NamedTuple):
    encoder: FakeArray
    decoder: FakeArray
    bias: FakeArray


class Optimizer(NamedTuple):
    m: Params
    v: Params
    step: FakeArray


class TrainState(NamedTuple):
    params: Params
    optimizer: Optimizer
    step: FakeArray
    tokens_seen: FakeArray
    fired_mask: FakeArray
    loss_ema: FakeArray
    initial_loss_ema: FakeArray
    ema_updates: FakeArray


def fake_state(condition: str, *, decoder_shape_delta: int = 0) -> TrainState:
    sae = FakeSAEConfig(condition)
    features = sae.total_features

    def params(dtype: str, *, altered: bool = False) -> Params:
        decoder_features = features + decoder_shape_delta if altered else features
        return Params(
            FakeArray((features, 2_816), dtype),
            FakeArray((2_816, decoder_features), dtype),
            FakeArray((features,), dtype),
        )

    return TrainState(
        params("bfloat16", altered=True),
        Optimizer(
            params("float32"),
            params("float32"),
            FakeArray((), "int32", scalar=1),
        ),
        FakeArray((), "int32", scalar=1),
        FakeArray((), "int32", scalar=16_384),
        FakeArray((features,), "bool"),
        FakeArray((), "float32"),
        FakeArray((), "float32"),
        FakeArray((), "int32", scalar=1),
    )


@dataclass(frozen=True)
class FakeTiming:
    steps: int = 1
    tokens: int = 16_384


@dataclass(frozen=True)
class FakeResult:
    state: object
    metrics: tuple[object, ...] = (object(),)
    timing: FakeTiming = FakeTiming()
    latest_checkpoint: str | None = None
    final_checkpoint: str | None = None


@dataclass(frozen=True)
class FakeMesh:
    shape: dict[str, int]


@dataclass(frozen=True)
class FakeDistributedConfig:
    data_axis: str = "data"
    feature_axis: str = "feature"
    microbatches_per_update: int = 4


def training_config(condition: str, *, capacity: int | None = None) -> FakeTrainingConfig:
    return FakeTrainingConfig(FakeSAEConfig(condition), a_capacity=capacity)


def campaign() -> SmokeCampaignIdentity:
    return SmokeCampaignIdentity(
        config_sha256=digest("1"),
        repository_content_sha256=digest("2"),
        train_capture_manifest_sha256=digest("3"),
        eval_capture_manifest_sha256=digest("4"),
        consumer_environment_sha256=digest("5"),
        capture_compatibility_sha256=digest("6"),
        full_shape=SAEFullShape.gemma4_26b_a4b(),
        checkpoint_payload_root="gs://smoke-bucket/checkpoints",
    )


def runner(tmp_path: Path) -> SAESmokeRunner:
    return SAESmokeRunner(
        tmp_path / "control",
        campaign(),
        smoke_cache_root=tmp_path / "local-cache",
    )


def direct_attempt(tmp_path: Path, condition: str, *, index: int = 1):
    selected = campaign()
    selected_runner = SAESmokeRunner(
        tmp_path / "control",
        selected,
        smoke_cache_root=tmp_path / "local-cache",
    )
    attempt_id = f"{index:032x}"
    paths = selected_runner.paths(
        condition, attempt_id, condition_attempt_number=1
    )
    shape = selected.full_shape.for_condition(condition)
    return SmokeAttemptIdentity(
        campaign_sha256=selected.sha256,
        condition=condition,
        condition_index=("F", "A", "B").index(condition),
        condition_attempt_number=1,
        attempt_id=attempt_id,
        full_shape_sha256=selected.full_shape.sha256,
        condition_shape=shape,
        cache_path=str(paths.cache_path),
        checkpoint_uri=paths.checkpoint_uri,
        expected_initial_step=0,
        expected_final_step=1,
        expected_initial_tokens_seen=0,
        expected_final_tokens_seen=16_384,
    )


def training_sha256(config: FakeTrainingConfig) -> str:
    encoded = json.dumps(
        asdict(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class FakeRuntime:
    def __init__(
        self,
        config: FakeTrainingConfig,
        mesh: FakeMesh,
        *,
        state: object | None = None,
    ) -> None:
        self.config = config
        self.mesh = mesh
        self.updated_state = state or fake_state(config.sae.condition)
        self.restored_state = fake_state(config.sae.condition)
        self.events: list[str] = []
        self.run_args: tuple[Any, ...] | None = None
        self.run_kwargs: dict[str, Any] | None = None
        self.saved_uri: str | None = None
        self.saved_state: object | None = None
        self.saved_metadata: dict[str, Any] | None = None
        self.inspection_transform = lambda value: value
        self.restore_metadata_transform = lambda value: value
        self.state_digest_values = [digest("a"), digest("a")]

    def run_training(self, *args, **kwargs):
        self.events.append("run")
        self.run_args = args
        self.run_kwargs = kwargs
        return FakeResult(self.updated_state)

    def save_checkpoint(self, uri, state, mesh, config, metadata=None):
        self.events.append("save")
        assert mesh is self.mesh
        assert config is self.config
        self.saved_uri = uri
        self.saved_state = state
        self.saved_metadata = dict(metadata or {})

    def inspection(self, uri: str) -> OrbaxPayloadInspection:
        self.events.append("inspect")
        assert uri == self.saved_uri
        assert self.saved_metadata is not None
        prefix = uri[5:].split("/", 1)[1]
        header = {
            "checkpoint_version": 1,
            "training_config": asdict(self.config),
            "training_config_sha256": training_sha256(self.config),
            "mesh_shape": self.mesh.shape,
            "metadata": self.saved_metadata,
        }
        value = OrbaxPayloadInspection(
            checkpoint_uri=uri,
            finalized=True,
            checkpoint_header=header,
            objects=(
                GCSObjectVersion(prefix + "/zarr/1", 12, 123, "crc-one"),
                GCSObjectVersion(prefix + "/_CHECKPOINT_METADATA", 11, 45, "crc-meta"),
            ),
        )
        return self.inspection_transform(value)

    def restore(self, uri, mesh, config, distributed):
        self.events.append("restore-fresh")
        assert uri == self.saved_uri
        assert mesh is self.mesh
        assert config is self.config
        assert distributed == FakeDistributedConfig()
        assert self.saved_metadata is not None
        return RestoredPayload(
            self.restored_state,
            self.restore_metadata_transform(dict(self.saved_metadata)),
        )

    def state_digest(self, state):
        self.events.append("state-digest")
        assert state in (self.updated_state, self.restored_state)
        return self.state_digest_values.pop(0)

    def progress(self, state):
        self.events.append("progress")
        assert state in (self.updated_state, self.restored_state)
        return StateProgress(1, 16_384, 1, 1)

    def consensus(self, value: str, label: str) -> str:
        self.events.append("consensus:" + label)
        return value

    def as_runtime(self) -> TPUSmokeRuntime:
        return TPUSmokeRuntime(
            run_training=self.run_training,
            save_checkpoint=self.save_checkpoint,
            inspect_payload=self.inspection,
            restore_fresh=self.restore,
            state_digest=self.state_digest,
            state_progress=self.progress,
            consensus_digest=self.consensus,
        )


def adapter(config: FakeTrainingConfig, fake: FakeRuntime) -> SAESmokeTPUAdapter:
    return SAESmokeTPUAdapter(
        campaign_identity=campaign(),
        training_config=config,
        batches="one-update-source",
        mean="training-mean",
        mesh=fake.mesh,
        distributed_config=FakeDistributedConfig(),
        smoke_a_capacity=config.a_capacity,
        runtime=fake.as_runtime(),
    )


def test_module_import_is_jax_orbax_and_gcs_free() -> None:
    script = """
import sys
import moe_sae.sae_smoke_tpu
assert 'jax' not in sys.modules
assert 'orbax' not in sys.modules
assert 'google.cloud.storage' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_full_f_smoke_uses_one_production_update_and_direct_payload_roundtrip(
    tmp_path: Path,
) -> None:
    config = training_config("F")
    mesh = FakeMesh({"data": 4, "feature": 8})
    fake = FakeRuntime(config, mesh)
    selected_adapter = adapter(config, fake)

    outcome = runner(tmp_path).run_attempt(
        "F", attempt_id="1" * 32, callbacks=selected_adapter.callbacks()
    )

    assert fake.run_args == (
        config,
        "one-update-source",
        "training-mean",
        mesh,
    )
    assert fake.run_kwargs == {
        "state": None,
        "distributed": FakeDistributedConfig(),
        "a_capacity": None,
        "checkpoint_dir": None,
        "checkpoint_payload_root": None,
        "stop_after_tokens": 16_384,
        "require_health_pass": False,
        "metric_callback": None,
        "process_local_inputs": True,
        "prior_optimizer_wall_seconds": 0.0,
        "prior_compile_seconds": 0.0,
        "prior_checkpoint_seconds": 0.0,
        "process_local_source_offset": 0,
        "checkpoint_identity": None,
    }
    assert fake.saved_uri == outcome.attempt.checkpoint_uri
    assert fake.saved_state is fake.updated_state
    assert fake.saved_metadata is not None
    assert fake.saved_metadata["smoke_campaign_identity"] == campaign().to_mapping()
    assert fake.saved_metadata["smoke_attempt_identity"] == outcome.attempt.to_mapping()
    assert fake.saved_metadata["smoke_attempt_sha256"] == outcome.attempt.sha256
    assert fake.saved_metadata["checkpoint_payload_uri"] == outcome.attempt.checkpoint_uri
    assert fake.saved_metadata["tokens_seen"] == 16_384
    assert fake.saved_metadata["steps"] == 1
    assert outcome.state_sha256 == digest("a")
    assert outcome.checkpoint_sha256 == checkpoint_inventory_sha256(
        outcome.attempt.checkpoint_uri,
        fake.inspection(outcome.attempt.checkpoint_uri).objects,
    )
    assert fake.events.count("inspect") == 3  # pre/post restore plus assertion above
    assert not (tmp_path / "control" / "training").exists()
    assert not (tmp_path / "control" / "runs").exists()
    assert not (tmp_path / "control" / "capacity").exists()
    assert not (tmp_path / "control" / "eval").exists()


def test_a_requires_and_passes_explicit_one_batch_capacity(tmp_path: Path) -> None:
    attempt = direct_attempt(tmp_path, "A")
    mesh = FakeMesh({"data": 4, "feature": 8})

    missing = training_config("A")
    missing_runtime = FakeRuntime(missing, mesh)
    with pytest.raises(ValueError, match="smoke A capacity"):
        adapter(missing, missing_runtime).update_once(attempt)
    assert missing_runtime.events == []

    config = training_config("A", capacity=2_048)
    fake = FakeRuntime(config, mesh)
    result = adapter(config, fake).update_once(attempt)
    assert result.final_tokens_seen == 16_384
    assert fake.run_kwargs is not None
    assert fake.run_kwargs["a_capacity"] == 2_048


def test_non_a_condition_rejects_capacity_before_training(tmp_path: Path) -> None:
    attempt = direct_attempt(tmp_path, "B")
    config = training_config("B", capacity=128)
    fake = FakeRuntime(config, FakeMesh({"data": 4, "feature": 8}))
    with pytest.raises(SAESmokeIntegrityError, match="cannot have an A capacity"):
        adapter(config, fake).update_once(attempt)
    assert fake.events == []


def test_nonproduction_config_and_wrong_parameter_shape_fail_closed(
    tmp_path: Path,
) -> None:
    attempt = direct_attempt(tmp_path, "F")
    wrong_config = replace(training_config("F"), effective_batch_tokens=8_192)
    wrong_runtime = FakeRuntime(wrong_config, FakeMesh({"data": 4, "feature": 8}))
    with pytest.raises(SAESmokeIntegrityError, match="effective_batch_tokens"):
        adapter(wrong_config, wrong_runtime).update_once(attempt)
    assert wrong_runtime.events == []

    config = training_config("F")
    bad_state = fake_state("F", decoder_shape_delta=-1)
    bad_runtime = FakeRuntime(
        config, FakeMesh({"data": 4, "feature": 8}), state=bad_state
    )
    with pytest.raises(SAESmokeIntegrityError, match="params.decoder shape"):
        adapter(config, bad_runtime).update_once(attempt)


def test_tampered_external_header_is_rejected_as_inspection_failure(
    tmp_path: Path,
) -> None:
    config = training_config("F")
    fake = FakeRuntime(config, FakeMesh({"data": 4, "feature": 8}))

    def tamper(value: OrbaxPayloadInspection) -> OrbaxPayloadInspection:
        header = dict(value.checkpoint_header)
        header["training_config_sha256"] = digest("f")
        return replace(value, checkpoint_header=header)

    fake.inspection_transform = tamper
    with pytest.raises(SAESmokeAttemptFailed, match="TrainingConfig digest") as caught:
        runner(tmp_path).run_attempt(
            "F", attempt_id="2" * 32, callbacks=adapter(config, fake).callbacks()
        )
    failure = json.loads(caught.value.failure_path.read_text(encoding="utf-8"))
    assert failure["stage"] == "external-checkpoint-inspection"


def test_restore_uses_fresh_path_and_revalidates_state_and_inventory(
    tmp_path: Path,
) -> None:
    config = training_config("F")
    fake = FakeRuntime(config, FakeMesh({"data": 4, "feature": 8}))
    fake.state_digest_values = [digest("a"), digest("b")]

    with pytest.raises(SAESmokeAttemptFailed, match="restored full TrainState digest") as caught:
        runner(tmp_path).run_attempt(
            "F", attempt_id="3" * 32, callbacks=adapter(config, fake).callbacks()
        )

    assert "restore-fresh" in fake.events
    assert fake.events.count("inspect") == 1
    failure = json.loads(caught.value.failure_path.read_text(encoding="utf-8"))
    assert failure["stage"] == "orbax-restore"


def test_inventory_change_after_restore_is_detected(tmp_path: Path) -> None:
    config = training_config("F")
    fake = FakeRuntime(config, FakeMesh({"data": 4, "feature": 8}))
    calls = 0

    def change_second(value: OrbaxPayloadInspection) -> OrbaxPayloadInspection:
        nonlocal calls
        calls += 1
        if calls == 2:
            objects = list(value.objects)
            objects[0] = replace(objects[0], generation=objects[0].generation + 1)
            return replace(value, objects=tuple(objects))
        return value

    fake.inspection_transform = change_second
    with pytest.raises(SAESmokeAttemptFailed, match="inventory changed"):
        runner(tmp_path).run_attempt(
            "F", attempt_id="4" * 32, callbacks=adapter(config, fake).callbacks()
        )


def test_inventory_digest_is_sorted_and_binds_generation_size_and_crc() -> None:
    uri = "gs://bucket/root/sae-smoke/campaign/checkpoint"
    prefix = "root/sae-smoke/campaign/checkpoint"
    objects = (
        GCSObjectVersion(prefix + "/b", 2, 20, "crc-b"),
        GCSObjectVersion(prefix + "/a", 1, 10, "crc-a"),
    )
    expected = checkpoint_inventory_sha256(uri, objects)
    assert checkpoint_inventory_sha256(uri, tuple(reversed(objects))) == expected
    for changed in (
        replace(objects[0], generation=3),
        replace(objects[0], size=21),
        replace(objects[0], crc32c="crc-other"),
    ):
        assert checkpoint_inventory_sha256(uri, (changed, objects[1])) != expected
    with pytest.raises(SAESmokeIntegrityError, match="duplicate"):
        checkpoint_inventory_sha256(
            uri,
            (objects[0], replace(objects[0], generation=3)),
        )


class Shard:
    def __init__(self, index: object, data: object) -> None:
        self.index = index
        self.data = data


def hash_array(
    value: object,
    *,
    index: tuple[object, ...] = (),
    replicas: int = 1,
    scalar: int | None = None,
) -> FakeArray:
    data = np.asarray(value)
    shards = tuple(Shard(index, data.copy()) for _ in range(replicas))
    return FakeArray(data.shape, str(data.dtype), scalar=scalar, addressable_shards=shards)


def hash_state(*, step: int = 1, reverse_encoder_shards: bool = False) -> TrainState:
    first = Shard((slice(0, 1), slice(0, 2)), np.asarray([[1.0, 2.0]], np.float32))
    duplicate = Shard((slice(0, 1), slice(0, 2)), np.asarray([[1.0, 2.0]], np.float32))
    second = Shard((slice(1, 2), slice(0, 2)), np.asarray([[3.0, 4.0]], np.float32))
    encoder_shards = (second, duplicate, first) if reverse_encoder_shards else (
        first,
        duplicate,
        second,
    )
    encoder = FakeArray((2, 2), "float32", addressable_shards=encoder_shards)

    def params() -> Params:
        return Params(
            encoder,
            hash_array(np.asarray([[5.0, 6.0], [7.0, 8.0]], np.float32)),
            hash_array(np.asarray([0.0, 0.0], np.float32)),
        )

    scalar_step = hash_array(np.asarray(step, np.int32), scalar=step, replicas=2)
    tokens = hash_array(np.asarray(16_384, np.int32), scalar=16_384, replicas=2)
    ema = hash_array(np.asarray(1, np.int32), scalar=1, replicas=2)
    return TrainState(
        params(),
        Optimizer(params(), params(), scalar_step),
        scalar_step,
        tokens,
        hash_array(np.asarray([True, False], bool)),
        hash_array(np.asarray(1.0, np.float32)),
        hash_array(np.asarray(2.0, np.float32)),
        ema,
    )


def scalar_reader(value: FakeArray) -> int:
    assert value.scalar is not None
    return value.scalar


def test_train_state_hash_orders_and_deduplicates_addressable_replicas() -> None:
    original = hash_state()
    first = hash_addressable_train_state(
        original, to_numpy=np.asarray, scalar_reader=scalar_reader
    )
    reordered = hash_addressable_train_state(
        hash_state(reverse_encoder_shards=True),
        to_numpy=np.asarray,
        scalar_reader=scalar_reader,
    )
    changed_progress = hash_addressable_train_state(
        hash_state(step=2), to_numpy=np.asarray, scalar_reader=scalar_reader
    )
    changed_decoder = hash_array(
        np.asarray([[5.0, 6.0], [7.0, 9.0]], np.float32)
    )
    changed_state = original._replace(
        params=original.params._replace(decoder=changed_decoder)
    )
    changed_parameter = hash_addressable_train_state(
        changed_state, to_numpy=np.asarray, scalar_reader=scalar_reader
    )
    assert first == reordered
    assert first != changed_progress
    assert first != changed_parameter


def test_train_state_hash_rejects_disagreeing_local_replicas() -> None:
    state = hash_state()
    encoder = state.params.encoder
    bad_shards = list(encoder.addressable_shards)
    bad_shards[1] = Shard(
        bad_shards[1].index,
        np.asarray([[99.0, 2.0]], np.float32),
    )
    bad_encoder = FakeArray(
        encoder.shape,
        encoder.dtype,
        addressable_shards=tuple(bad_shards),
    )
    bad_state = state._replace(params=state.params._replace(encoder=bad_encoder))
    with pytest.raises(SAESmokeIntegrityError, match="replicas disagree"):
        hash_addressable_train_state(
            bad_state, to_numpy=np.asarray, scalar_reader=scalar_reader
        )


def test_host_digest_consensus_fails_closed() -> None:
    assert require_identical_host_digests(
        digest("a"), (digest("a"),) * 4, label="TrainState"
    ) == digest("a")
    with pytest.raises(SAESmokeIntegrityError, match="hosts disagree"):
        require_identical_host_digests(
            digest("a"),
            (digest("a"), digest("a"), digest("b"), digest("a")),
            label="TrainState",
        )
