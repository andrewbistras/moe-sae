"""Controller-owned semantic authority for DeepSeek-R1 MFU V2 evidence.

The numerical reducer in :mod:`moe_sae.r1.mfu_v2` intentionally knows
nothing about GCS, Xprof, HLO protobufs, ArrayRecord, or writer receipts.  This
module is the trust boundary that turns those opaque artifacts into facts:

* every byte read is pinned to a declared object generation and content hash;
* policy records pin claimed authority, resolver, and parser identities;
* protocol checks exercise preregistration and compile/warmup chronology;
* versioned parser outputs are compared byte-for-byte with published receipts;
* protocol-oracle reduction remains available for adversarial tests, but it can
  serialize only an explicitly UNVERIFIED qualification;
* headline PASS sealing and report consumption remain disabled until an
  operator-owned closed factory supplies exact production parsers and an
  independently replayed controller-admission proof.

No permissive/default parser exists.  A deployment without real parsers for its
exact raw formats cannot instantiate a usable production authority and fails
closed.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, fields as dataclass_fields
import hashlib
import math
import time
from typing import Any, Callable, ClassVar, Mapping, Protocol, Self, Sequence

from .control_store import JSONCASStore
from .flops import (
    DEEPSEEK_R1_PREFIX_GEOMETRY_V1,
    deepseek_r1_prefix_flop_coefficients,
)
from .image_protocol import DeviceInventoryV1
from .mfu_v2 import (
    MEDIA_FORMULA_RECONCILIATION_RECEIPT_V2,
    MEDIA_INPUT_REDUCTION_RECEIPT_V2,
    MEDIA_PREDECLARATION_RECEIPT_V2,
    MEDIA_SEMANTIC_AUTHORITY_V2,
    MEDIA_WARMUP_RECEIPT_V2,
    MEDIA_WINDOW_REDUCTION_RECEIPT_V2,
    MFUSemanticReceiptsV2,
    MFUV2ContractError,
    MFUWindowPlanEntryV2,
    PackedInputSliceV2,
    PrefixFlopFormulaV2,
    PrefixMFUQualificationV2,
    PrefixMFURunContractV2,
    PrefixMFUWindowEvidenceV2,
    qualify_prefix_mfu_v2,
)
from .protocol import (
    CanonicalRecord,
    ContentRefV1,
    IDENTIFIER_PATTERN,
    ProtocolError,
    RegionalReplicaV1,
    SHA256_PATTERN,
    canonical_sha256,
    parse_json,
    require_integer,
    require_record_object,
    require_string,
)


MEDIA_AUTHORITY_POLICY_V2 = "application/vnd.moe-sae.r1.mfu-authority-policy.v2+json"
MEDIA_CONTENT_RESOLVER_IDENTITY_V2 = (
    "application/vnd.moe-sae.r1.content-resolver-identity.v2+json"
)
MEDIA_FORMULA_PARSER_IDENTITY_V2 = (
    "application/vnd.moe-sae.r1.formula-parser-identity.v2+json"
)
MEDIA_INPUT_PARSER_IDENTITY_V2 = (
    "application/vnd.moe-sae.r1.input-parser-identity.v2+json"
)
MEDIA_WINDOW_PARSER_IDENTITY_V2 = (
    "application/vnd.moe-sae.r1.window-parser-identity.v2+json"
)
MEDIA_VERIFIED_QUALIFICATION_ENVELOPE_V2 = (
    "application/vnd.moe-sae.r1.verified-mfu-qualification.v2+json"
)
_MAX_IN_MEMORY_RESOLUTION_CACHE_BYTES = 8 * 1024 * 1024
_PRODUCTION_DISABLED_MESSAGE = (
    "production MFU V2 PASS is disabled until an operator-owned closed factory "
    "binds exact real parsers and independently verifies the immutable "
    "controller admission event"
)


class MFUAuthorityError(MFUV2ContractError):
    """Authority policy, content, receipt, or parser evidence failed closed."""


def _identifier(value: Any, name: str) -> str:
    return require_string(value, name, pattern=IDENTIFIER_PATTERN)


def _sha(value: Any, name: str) -> str:
    return require_string(value, name, pattern=SHA256_PATTERN)


def _content(value: Any, name: str, media_type: str | None = None) -> ContentRefV1:
    if not isinstance(value, ContentRefV1):
        raise MFUAuthorityError(f"{name} must be ContentRefV1")
    if media_type is not None and value.media_type != media_type:
        raise MFUAuthorityError(f"{name} media_type must be exactly {media_type!r}")
    return value


def _same_payload(left: ContentRefV1, right: ContentRefV1) -> bool:
    return (
        left.sha256,
        left.size_bytes,
        left.media_type,
    ) == (
        right.sha256,
        right.size_bytes,
        right.media_type,
    )


def _wire(value: Any) -> Any:
    if isinstance(value, CanonicalRecord):
        return value.as_dict()
    if isinstance(value, tuple):
        return [_wire(item) for item in value]
    if value is None or isinstance(value, (str, int)):
        return value
    raise MFUAuthorityError(f"unsupported wire value: {type(value).__name__}")


def _record_dict(record: CanonicalRecord) -> dict[str, Any]:
    return {
        "schema": record.SCHEMA,
        **{
            item.name: _wire(getattr(record, item.name))
            for item in dataclass_fields(record)
            if item.init and not item.name.startswith("_")
        },
    }


def _record_object(value: Any, record_type: type[CanonicalRecord]) -> Mapping[str, Any]:
    names = tuple(
        item.name
        for item in dataclass_fields(record_type)
        if item.init and not item.name.startswith("_")
    )
    return require_record_object(value, schema=record_type.SCHEMA, fields=names)


def _content_identity(value: ContentRefV1) -> dict[str, Any]:
    return {
        "sha256": value.sha256,
        "size_bytes": value.size_bytes,
        "media_type": value.media_type,
    }


def _logical_projection(value: Any) -> Any:
    if isinstance(value, ContentRefV1):
        return _content_identity(value)
    if isinstance(value, tuple):
        return [_logical_projection(item) for item in value]
    if value is None or isinstance(value, (str, int)):
        return value
    raise MFUAuthorityError(
        f"unsupported logical projection value: {type(value).__name__}"
    )


def window_observation_digest_v2(
    evidence: PrefixMFUWindowEvidenceV2,
) -> str:
    """Hash every producer projection while ignoring replica-list metadata."""

    if not isinstance(evidence, PrefixMFUWindowEvidenceV2):
        raise MFUAuthorityError("evidence must be PrefixMFUWindowEvidenceV2")
    projection = {
        "schema": "moe-sae.r1.window-observation-projection.v2",
        **{
            item.name: _logical_projection(getattr(evidence, item.name))
            for item in dataclass_fields(evidence)
            if item.init and not item.name.startswith("_")
        },
    }
    return canonical_sha256(projection)


@dataclass(frozen=True, slots=True)
class TrustedMFUAuthorityPolicyV2(CanonicalRecord):
    """Externally provisioned trust root; candidate runs cannot create trust."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.trusted-mfu-authority-policy.v2"

    policy_id: str
    authority_identity: ContentRefV1
    resolver_identity: ContentRefV1
    formula_parser_identity: ContentRefV1
    input_parser_identity: ContentRefV1
    window_parser_identity: ContentRefV1
    controller_receipt_prefix: str
    allowed_replica_regions: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.policy_id, "policy_id")
        for reference, name, media in (
            (
                self.authority_identity,
                "authority_identity",
                MEDIA_SEMANTIC_AUTHORITY_V2,
            ),
            (
                self.resolver_identity,
                "resolver_identity",
                MEDIA_CONTENT_RESOLVER_IDENTITY_V2,
            ),
            (
                self.formula_parser_identity,
                "formula_parser_identity",
                MEDIA_FORMULA_PARSER_IDENTITY_V2,
            ),
            (
                self.input_parser_identity,
                "input_parser_identity",
                MEDIA_INPUT_PARSER_IDENTITY_V2,
            ),
            (
                self.window_parser_identity,
                "window_parser_identity",
                MEDIA_WINDOW_PARSER_IDENTITY_V2,
            ),
        ):
            _content(reference, name, media)
        prefix = require_string(
            self.controller_receipt_prefix, "controller_receipt_prefix"
        )
        if not prefix.startswith("gs://") or not prefix.endswith("/"):
            raise MFUAuthorityError(
                "controller_receipt_prefix must be an absolute gs:// prefix"
            )
        bucket, separator, name = prefix[5:].partition("/")
        if (
            not separator
            or not bucket
            or not name
            or "//" in prefix[5:]
            or any(part in {".", ".."} for part in prefix[5:].split("/"))
        ):
            raise MFUAuthorityError("controller_receipt_prefix is not normalized")
        regions = tuple(self.allowed_replica_regions)
        if not regions or any(
            _identifier(region, "allowed_replica_regions[]") != region
            for region in regions
        ):
            raise MFUAuthorityError("allowed_replica_regions must be non-empty")
        if regions != tuple(sorted(set(regions))):
            raise MFUAuthorityError("allowed_replica_regions must be unique and sorted")
        object.__setattr__(self, "allowed_replica_regions", regions)

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        regions = obj["allowed_replica_regions"]
        if not isinstance(regions, list):
            raise MFUAuthorityError("allowed_replica_regions must be an array")
        return cls(
            policy_id=_identifier(obj["policy_id"], "policy_id"),
            authority_identity=ContentRefV1.from_dict(obj["authority_identity"]),
            resolver_identity=ContentRefV1.from_dict(obj["resolver_identity"]),
            formula_parser_identity=ContentRefV1.from_dict(
                obj["formula_parser_identity"]
            ),
            input_parser_identity=ContentRefV1.from_dict(obj["input_parser_identity"]),
            window_parser_identity=ContentRefV1.from_dict(
                obj["window_parser_identity"]
            ),
            controller_receipt_prefix=require_string(
                obj["controller_receipt_prefix"], "controller_receipt_prefix"
            ),
            allowed_replica_regions=tuple(
                _identifier(item, "allowed_replica_regions[]") for item in regions
            ),
        )


@dataclass(frozen=True, slots=True)
class ResolvedContentV2:
    """Bytes fetched from one exact replica generation."""

    reference: ContentRefV1
    replica: RegionalReplicaV1
    value: bytes

    def __post_init__(self) -> None:
        _content(self.reference, "reference")
        if not isinstance(self.replica, RegionalReplicaV1):
            raise MFUAuthorityError("replica must be RegionalReplicaV1")
        if self.replica not in self.reference.replicas:
            raise MFUAuthorityError("resolved replica is not declared by reference")
        if not isinstance(self.value, bytes):
            raise MFUAuthorityError("resolved value must be bytes")
        if len(self.value) != self.reference.size_bytes:
            raise MFUAuthorityError("resolved byte size does not match content ref")
        if hashlib.sha256(self.value).hexdigest() != self.reference.sha256:
            raise MFUAuthorityError("resolved SHA-256 does not match content ref")
        if _crc32c_base64(self.value) != self.replica.crc32c:
            raise MFUAuthorityError("resolved CRC32C does not match replica metadata")


class GenerationPinnedContentResolverV2(Protocol):
    @property
    def identity(self) -> ContentRefV1: ...

    def resolve(self, reference: ContentRefV1) -> ResolvedContentV2: ...


class JSONCASGenerationPinnedResolverV2:
    """Exact-generation resolver for canonical objects in a JSON CAS store."""

    def __init__(self, store: JSONCASStore, *, identity: ContentRefV1) -> None:
        self._store = store
        self._identity = _content(
            identity, "identity", MEDIA_CONTENT_RESOLVER_IDENTITY_V2
        )

    @property
    def identity(self) -> ContentRefV1:
        return self._identity

    def resolve(self, reference: ContentRefV1) -> ResolvedContentV2:
        _content(reference, "reference")
        for replica in reference.replicas:
            version = self._store.read(replica.uri)
            if version is None or version.generation != replica.generation:
                continue
            if version.sha256 != reference.sha256:
                raise MFUAuthorityError(
                    "generation-pinned store object has the wrong SHA-256"
                )
            return ResolvedContentV2(reference, replica, version.value)
        raise MFUAuthorityError("no declared replica resolves at its exact generation")


def _gcs_object(reference: RegionalReplicaV1) -> tuple[str, str]:
    bucket, separator, name = reference.uri[5:].partition("/")
    if (
        not separator
        or not bucket
        or not name
        or name.endswith("/")
        or "//" in name
        or any(part in {"", ".", ".."} for part in name.split("/"))
    ):
        raise MFUAuthorityError("replica does not name a normalized GCS object")
    return bucket, name


class GCSGenerationPinnedContentResolverV2:
    """Exact-generation GCS resolver that hashes downloaded bytes.

    This intentionally exposes no listing operation.  Format parsers receive
    only objects named by the immutable evidence graph.  This reference
    implementation materializes one object at a time; deployments with
    multi-GiB traces should inject a streaming resolver/parser implementation
    before production qualification.
    """

    def __init__(
        self,
        *,
        identity: ContentRefV1,
        client: Any | None = None,
        project: str | None = None,
        timeout_seconds: float = 1_800.0,
    ) -> None:
        self._identity = _content(
            identity, "identity", MEDIA_CONTENT_RESOLVER_IDENTITY_V2
        )
        if isinstance(timeout_seconds, bool) or not isinstance(
            timeout_seconds, (int, float)
        ):
            raise MFUAuthorityError("timeout_seconds must be positive")
        self._timeout = float(timeout_seconds)
        if not math.isfinite(self._timeout) or self._timeout <= 0:
            raise MFUAuthorityError("timeout_seconds must be positive")
        if client is None:
            try:
                from google.cloud import storage
            except ImportError as error:
                raise MFUAuthorityError(
                    "GCS resolution requires google-cloud-storage (`--extra tpu`)"
                ) from error
            client = storage.Client(project=project)
        self._client = client

    @property
    def identity(self) -> ContentRefV1:
        return self._identity

    def resolve(self, reference: ContentRefV1) -> ResolvedContentV2:
        _content(reference, "reference")
        last_error: Exception | None = None
        for replica in reference.replicas:
            bucket_name, object_name = _gcs_object(replica)
            blob = self._client.bucket(bucket_name).blob(object_name)
            try:
                blob.reload(
                    if_generation_match=replica.generation,
                    timeout=self._timeout,
                )
                if (
                    int(blob.generation or 0) != replica.generation
                    or int(blob.size or 0) != replica.size_bytes
                    or str(blob.crc32c or "") != replica.crc32c
                ):
                    raise MFUAuthorityError(
                        "GCS replica metadata differs at the pinned generation"
                    )
                value = blob.download_as_bytes(
                    if_generation_match=replica.generation,
                    checksum="crc32c",
                    timeout=self._timeout,
                )
                if type(value) is not bytes:
                    raise MFUAuthorityError("GCS resolver returned non-bytes content")
                return ResolvedContentV2(reference, replica, value)
            except MFUAuthorityError:
                raise
            except Exception as error:
                last_error = error
                continue
        raise MFUAuthorityError(
            "no declared GCS replica resolves at its exact generation"
        ) from last_error


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    try:
        return require_integer(value, name, minimum=minimum)
    except ProtocolError as error:
        raise MFUAuthorityError(str(error)) from error


def _receipt_ref_digest(references: Sequence[ContentRefV1]) -> str:
    return canonical_sha256(
        {
            "schema": "moe-sae.r1.mfu-artifact-bundle.v2",
            "artifacts": [_content_identity(item) for item in references],
        }
    )


def _crc32c_base64(value: bytes) -> str:
    """Return GCS's big-endian base64 CRC32C without an import-time dependency."""

    try:
        import google_crc32c
    except ImportError:
        pass
    else:
        return base64.b64encode(google_crc32c.value(value).to_bytes(4, "big")).decode(
            "ascii"
        )
    crc = 0xFFFFFFFF
    for octet in value:
        crc ^= octet
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return base64.b64encode(
        ((crc ^ 0xFFFFFFFF) & 0xFFFFFFFF).to_bytes(4, "big")
    ).decode("ascii")


def _device_ids_digest(inventory: DeviceInventoryV1) -> str:
    return canonical_sha256(
        {
            "schema": "moe-sae.r1.mfu-device-ids.v2",
            "device_ids": sorted(
                device_id
                for host in inventory.hosts
                for device_id in host.jax_global_device_ids
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class MFUPredeclarationReceiptPayloadV2(CanonicalRecord):
    """Controller-owned fence proving the complete plan predated measurement."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.mfu-predeclaration-receipt.v2"

    policy_sha256: str
    authority_sha256: str
    campaign_id: str
    campaign_head_uri: str
    campaign_head_generation: int
    campaign_head_revision: int
    work_id: str
    attempt_id: str
    executor_id: str
    scientific_identity_sha256: str
    deployment_identity_sha256: str
    run_contract_sha256: str
    window_plan_sha256: str
    controller_fence: int
    lease_acquired_at_unix_ns: int
    lease_expires_at_unix_ns: int
    registered_at_unix_ns: int
    measurement_not_before_unix_ns: int

    def __post_init__(self) -> None:
        _sha(self.policy_sha256, "policy_sha256")
        _sha(self.authority_sha256, "authority_sha256")
        _identifier(self.campaign_id, "campaign_id")
        uri = require_string(self.campaign_head_uri, "campaign_head_uri")
        if not uri.startswith("gs://"):
            raise MFUAuthorityError("campaign_head_uri must be a gs:// URI")
        _integer(self.campaign_head_generation, "campaign_head_generation", minimum=1)
        _integer(self.campaign_head_revision, "campaign_head_revision")
        _identifier(self.work_id, "work_id")
        _identifier(self.attempt_id, "attempt_id")
        _identifier(self.executor_id, "executor_id")
        _sha(self.scientific_identity_sha256, "scientific_identity_sha256")
        _sha(self.deployment_identity_sha256, "deployment_identity_sha256")
        _sha(self.run_contract_sha256, "run_contract_sha256")
        _sha(self.window_plan_sha256, "window_plan_sha256")
        _integer(self.controller_fence, "controller_fence", minimum=1)
        acquired = _integer(
            self.lease_acquired_at_unix_ns,
            "lease_acquired_at_unix_ns",
            minimum=1,
        )
        expires = _integer(
            self.lease_expires_at_unix_ns,
            "lease_expires_at_unix_ns",
            minimum=1,
        )
        registered = _integer(
            self.registered_at_unix_ns, "registered_at_unix_ns", minimum=1
        )
        not_before = _integer(
            self.measurement_not_before_unix_ns,
            "measurement_not_before_unix_ns",
            minimum=1,
        )
        if not (acquired <= registered <= not_before < expires):
            raise MFUAuthorityError(
                "preregistration and measurement fence must lie within the lease"
            )

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        return cls(
            policy_sha256=_sha(obj["policy_sha256"], "policy_sha256"),
            authority_sha256=_sha(obj["authority_sha256"], "authority_sha256"),
            campaign_id=_identifier(obj["campaign_id"], "campaign_id"),
            campaign_head_uri=require_string(
                obj["campaign_head_uri"], "campaign_head_uri"
            ),
            campaign_head_generation=_integer(
                obj["campaign_head_generation"], "campaign_head_generation", minimum=1
            ),
            campaign_head_revision=_integer(
                obj["campaign_head_revision"], "campaign_head_revision"
            ),
            work_id=_identifier(obj["work_id"], "work_id"),
            attempt_id=_identifier(obj["attempt_id"], "attempt_id"),
            executor_id=_identifier(obj["executor_id"], "executor_id"),
            scientific_identity_sha256=_sha(
                obj["scientific_identity_sha256"],
                "scientific_identity_sha256",
            ),
            deployment_identity_sha256=_sha(
                obj["deployment_identity_sha256"],
                "deployment_identity_sha256",
            ),
            run_contract_sha256=_sha(obj["run_contract_sha256"], "run_contract_sha256"),
            window_plan_sha256=_sha(obj["window_plan_sha256"], "window_plan_sha256"),
            controller_fence=_integer(
                obj["controller_fence"], "controller_fence", minimum=1
            ),
            lease_acquired_at_unix_ns=_integer(
                obj["lease_acquired_at_unix_ns"],
                "lease_acquired_at_unix_ns",
                minimum=1,
            ),
            lease_expires_at_unix_ns=_integer(
                obj["lease_expires_at_unix_ns"],
                "lease_expires_at_unix_ns",
                minimum=1,
            ),
            registered_at_unix_ns=_integer(
                obj["registered_at_unix_ns"], "registered_at_unix_ns", minimum=1
            ),
            measurement_not_before_unix_ns=_integer(
                obj["measurement_not_before_unix_ns"],
                "measurement_not_before_unix_ns",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class MFUCompileWarmupReceiptPayloadV2(CanonicalRecord):
    """Parsed compile and warmup chronology for one exact executable."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.mfu-compile-warmup-receipt.v2"

    run_contract_sha256: str
    attempt_id: str
    measurement_execution_id: str
    executable_sha256: str
    hlo_sha256: str
    device_inventory_sha256: str
    controller_fence: int
    compile_started_at_unix_ns: int
    compile_finished_at_unix_ns: int
    warmup_started_at_unix_ns: int
    warmup_finished_at_unix_ns: int
    completed_warmup_steps: int
    first_measurement_step: int
    first_measurement_started_at_unix_ns: int
    measurement_recompilations: int

    def __post_init__(self) -> None:
        _sha(self.run_contract_sha256, "run_contract_sha256")
        _identifier(self.attempt_id, "attempt_id")
        _identifier(self.measurement_execution_id, "measurement_execution_id")
        for name in (
            "executable_sha256",
            "hlo_sha256",
            "device_inventory_sha256",
        ):
            _sha(getattr(self, name), name)
        _integer(self.controller_fence, "controller_fence", minimum=1)
        chronology = tuple(
            _integer(getattr(self, name), name, minimum=1)
            for name in (
                "compile_started_at_unix_ns",
                "compile_finished_at_unix_ns",
                "warmup_started_at_unix_ns",
                "warmup_finished_at_unix_ns",
                "first_measurement_started_at_unix_ns",
            )
        )
        compile_started, compile_finished, warmup_started, warmup_finished, measured = (
            chronology
        )
        if (
            compile_started >= compile_finished
            or compile_finished > warmup_started
            or warmup_started >= warmup_finished
            or warmup_finished > measured
        ):
            raise MFUAuthorityError(
                "compile, warmup, and measurement chronology overlaps"
            )
        _integer(self.completed_warmup_steps, "completed_warmup_steps", minimum=1)
        _integer(self.first_measurement_step, "first_measurement_step")
        _integer(self.measurement_recompilations, "measurement_recompilations")

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        return cls(
            run_contract_sha256=_sha(obj["run_contract_sha256"], "run_contract_sha256"),
            attempt_id=_identifier(obj["attempt_id"], "attempt_id"),
            measurement_execution_id=_identifier(
                obj["measurement_execution_id"], "measurement_execution_id"
            ),
            executable_sha256=_sha(obj["executable_sha256"], "executable_sha256"),
            hlo_sha256=_sha(obj["hlo_sha256"], "hlo_sha256"),
            device_inventory_sha256=_sha(
                obj["device_inventory_sha256"], "device_inventory_sha256"
            ),
            controller_fence=_integer(
                obj["controller_fence"], "controller_fence", minimum=1
            ),
            compile_started_at_unix_ns=_integer(
                obj["compile_started_at_unix_ns"],
                "compile_started_at_unix_ns",
                minimum=1,
            ),
            compile_finished_at_unix_ns=_integer(
                obj["compile_finished_at_unix_ns"],
                "compile_finished_at_unix_ns",
                minimum=1,
            ),
            warmup_started_at_unix_ns=_integer(
                obj["warmup_started_at_unix_ns"], "warmup_started_at_unix_ns", minimum=1
            ),
            warmup_finished_at_unix_ns=_integer(
                obj["warmup_finished_at_unix_ns"],
                "warmup_finished_at_unix_ns",
                minimum=1,
            ),
            completed_warmup_steps=_integer(
                obj["completed_warmup_steps"], "completed_warmup_steps", minimum=1
            ),
            first_measurement_step=_integer(
                obj["first_measurement_step"], "first_measurement_step"
            ),
            first_measurement_started_at_unix_ns=_integer(
                obj["first_measurement_started_at_unix_ns"],
                "first_measurement_started_at_unix_ns",
                minimum=1,
            ),
            measurement_recompilations=_integer(
                obj["measurement_recompilations"], "measurement_recompilations"
            ),
        )


@dataclass(frozen=True, slots=True)
class MFUFormulaReconciliationReceiptPayloadV2(CanonicalRecord):
    """Parser result binding HLO operations to the one analytic coefficient ledger."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.mfu-formula-reconciliation.v2"

    run_contract_sha256: str
    formula_sha256: str
    executable_sha256: str
    hlo_sha256: str
    tensor_inventory_sha256: str
    operation_inventory_sha256: str
    analytic_formula_sha256: str
    coefficient_ledger_sha256: str
    sequences_per_step: int
    sequence_length: int
    counted_matmul_dtype: str
    denominator_dtype: str

    def __post_init__(self) -> None:
        for name in (
            "run_contract_sha256",
            "formula_sha256",
            "executable_sha256",
            "hlo_sha256",
            "tensor_inventory_sha256",
            "operation_inventory_sha256",
            "analytic_formula_sha256",
            "coefficient_ledger_sha256",
        ):
            _sha(getattr(self, name), name)
        _integer(self.sequences_per_step, "sequences_per_step", minimum=1)
        _integer(self.sequence_length, "sequence_length", minimum=1)
        if self.counted_matmul_dtype != "bfloat16":
            raise MFUAuthorityError("formula receipt must count bfloat16 matmuls")
        if self.denominator_dtype != "bfloat16":
            raise MFUAuthorityError("formula receipt denominator must be bfloat16")

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        return cls(
            **{
                name: _sha(obj[name], name)
                for name in (
                    "run_contract_sha256",
                    "formula_sha256",
                    "executable_sha256",
                    "hlo_sha256",
                    "tensor_inventory_sha256",
                    "operation_inventory_sha256",
                    "analytic_formula_sha256",
                    "coefficient_ledger_sha256",
                )
            },
            sequences_per_step=_integer(
                obj["sequences_per_step"], "sequences_per_step", minimum=1
            ),
            sequence_length=_integer(
                obj["sequence_length"], "sequence_length", minimum=1
            ),
            counted_matmul_dtype=require_string(
                obj["counted_matmul_dtype"], "counted_matmul_dtype"
            ),
            denominator_dtype=require_string(
                obj["denominator_dtype"], "denominator_dtype"
            ),
        )


@dataclass(frozen=True, slots=True)
class MFUInputReductionReceiptPayloadV2(CanonicalRecord):
    """Exact manifest/segment-ledger reduction for one predeclared slice."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.mfu-input-reduction.v2"

    run_contract_sha256: str
    input_slice_sha256: str
    parent_input_manifest_sha256: str
    segment_ledger_sha256: str
    group_index: int
    data_kind: str
    batch_start: int
    batch_end: int
    packed_sequence_count: int
    histogram_sha256: str
    valid_tokens: int
    causal_pairs: int

    def __post_init__(self) -> None:
        for name in (
            "run_contract_sha256",
            "input_slice_sha256",
            "parent_input_manifest_sha256",
            "segment_ledger_sha256",
            "histogram_sha256",
        ):
            _sha(getattr(self, name), name)
        _integer(self.group_index, "group_index")
        if self.data_kind not in {"REAL", "SYNTHETIC"}:
            raise MFUAuthorityError("data_kind must be REAL or SYNTHETIC")
        start = _integer(self.batch_start, "batch_start")
        end = _integer(self.batch_end, "batch_end", minimum=1)
        if end <= start:
            raise MFUAuthorityError("input reduction batch range must be non-empty")
        for name in ("packed_sequence_count", "valid_tokens", "causal_pairs"):
            _integer(getattr(self, name), name, minimum=1)

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        return cls(
            run_contract_sha256=_sha(obj["run_contract_sha256"], "run_contract_sha256"),
            input_slice_sha256=_sha(obj["input_slice_sha256"], "input_slice_sha256"),
            parent_input_manifest_sha256=_sha(
                obj["parent_input_manifest_sha256"], "parent_input_manifest_sha256"
            ),
            segment_ledger_sha256=_sha(
                obj["segment_ledger_sha256"], "segment_ledger_sha256"
            ),
            group_index=_integer(obj["group_index"], "group_index"),
            data_kind=require_string(obj["data_kind"], "data_kind"),
            batch_start=_integer(obj["batch_start"], "batch_start"),
            batch_end=_integer(obj["batch_end"], "batch_end", minimum=1),
            packed_sequence_count=_integer(
                obj["packed_sequence_count"], "packed_sequence_count", minimum=1
            ),
            histogram_sha256=_sha(obj["histogram_sha256"], "histogram_sha256"),
            valid_tokens=_integer(obj["valid_tokens"], "valid_tokens", minimum=1),
            causal_pairs=_integer(obj["causal_pairs"], "causal_pairs", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class MFUWindowReductionReceiptPayloadV2(CanonicalRecord):
    """Full trace/persistence/health/capacity parser result for one window."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.mfu-window-reduction.v2"

    run_contract_sha256: str
    attempt_id: str
    measurement_execution_id: str
    window_id: str
    input_slice_sha256: str
    executable_sha256: str
    hlo_sha256: str
    device_inventory_sha256: str
    controller_fence: int
    device_ids_sha256: str
    artifact_bundle_sha256: str
    observation_sha256: str
    step_start: int
    step_end: int
    measurement_started_at_unix_ns: int
    measurement_finished_at_unix_ns: int
    capacity_steps: int
    capacity_router_invocations: int
    capacity_routed_experts: int
    capacity_cell_count: int
    capacity_matrix_sha256: str
    maximum_capacity_numerator: int
    maximum_capacity_denominator: int
    overloaded_capacity_cells: int

    def __post_init__(self) -> None:
        for name in (
            "run_contract_sha256",
            "input_slice_sha256",
            "executable_sha256",
            "hlo_sha256",
            "device_inventory_sha256",
            "device_ids_sha256",
            "artifact_bundle_sha256",
            "observation_sha256",
            "capacity_matrix_sha256",
        ):
            _sha(getattr(self, name), name)
        _identifier(self.attempt_id, "attempt_id")
        _identifier(self.measurement_execution_id, "measurement_execution_id")
        _identifier(self.window_id, "window_id")
        _integer(self.controller_fence, "controller_fence", minimum=1)
        start = _integer(self.step_start, "step_start")
        end = _integer(self.step_end, "step_end", minimum=1)
        if end <= start:
            raise MFUAuthorityError("window reduction step range must be non-empty")
        began = _integer(
            self.measurement_started_at_unix_ns,
            "measurement_started_at_unix_ns",
            minimum=1,
        )
        finished = _integer(
            self.measurement_finished_at_unix_ns,
            "measurement_finished_at_unix_ns",
            minimum=1,
        )
        if finished <= began:
            raise MFUAuthorityError("window measurement interval must be non-empty")
        capacity_steps = _integer(self.capacity_steps, "capacity_steps", minimum=1)
        router_invocations = _integer(
            self.capacity_router_invocations,
            "capacity_router_invocations",
            minimum=1,
        )
        routed_experts = _integer(
            self.capacity_routed_experts,
            "capacity_routed_experts",
            minimum=1,
        )
        cell_count = _integer(
            self.capacity_cell_count, "capacity_cell_count", minimum=1
        )
        if capacity_steps * router_invocations * routed_experts != cell_count:
            raise MFUAuthorityError(
                "capacity dimensions do not multiply to capacity_cell_count"
            )
        numerator = _integer(
            self.maximum_capacity_numerator,
            "maximum_capacity_numerator",
            minimum=1,
        )
        denominator = _integer(
            self.maximum_capacity_denominator,
            "maximum_capacity_denominator",
            minimum=1,
        )
        overloaded = _integer(
            self.overloaded_capacity_cells, "overloaded_capacity_cells"
        )
        if overloaded > self.capacity_cell_count:
            raise MFUAuthorityError(
                "overloaded_capacity_cells exceeds capacity_cell_count"
            )
        if numerator * 10 > denominator * 11:
            raise MFUAuthorityError("a per-expert capacity cell exceeds 1.10")

    def as_dict(self) -> dict[str, Any]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        obj = _record_object(value, cls)
        return cls(
            run_contract_sha256=_sha(obj["run_contract_sha256"], "run_contract_sha256"),
            attempt_id=_identifier(obj["attempt_id"], "attempt_id"),
            measurement_execution_id=_identifier(
                obj["measurement_execution_id"], "measurement_execution_id"
            ),
            window_id=_identifier(obj["window_id"], "window_id"),
            input_slice_sha256=_sha(obj["input_slice_sha256"], "input_slice_sha256"),
            executable_sha256=_sha(obj["executable_sha256"], "executable_sha256"),
            hlo_sha256=_sha(obj["hlo_sha256"], "hlo_sha256"),
            device_inventory_sha256=_sha(
                obj["device_inventory_sha256"], "device_inventory_sha256"
            ),
            controller_fence=_integer(
                obj["controller_fence"], "controller_fence", minimum=1
            ),
            device_ids_sha256=_sha(obj["device_ids_sha256"], "device_ids_sha256"),
            artifact_bundle_sha256=_sha(
                obj["artifact_bundle_sha256"], "artifact_bundle_sha256"
            ),
            observation_sha256=_sha(obj["observation_sha256"], "observation_sha256"),
            step_start=_integer(obj["step_start"], "step_start"),
            step_end=_integer(obj["step_end"], "step_end", minimum=1),
            measurement_started_at_unix_ns=_integer(
                obj["measurement_started_at_unix_ns"],
                "measurement_started_at_unix_ns",
                minimum=1,
            ),
            measurement_finished_at_unix_ns=_integer(
                obj["measurement_finished_at_unix_ns"],
                "measurement_finished_at_unix_ns",
                minimum=1,
            ),
            capacity_cell_count=_integer(
                obj["capacity_cell_count"], "capacity_cell_count", minimum=1
            ),
            capacity_steps=_integer(obj["capacity_steps"], "capacity_steps", minimum=1),
            capacity_router_invocations=_integer(
                obj["capacity_router_invocations"],
                "capacity_router_invocations",
                minimum=1,
            ),
            capacity_routed_experts=_integer(
                obj["capacity_routed_experts"],
                "capacity_routed_experts",
                minimum=1,
            ),
            capacity_matrix_sha256=_sha(
                obj["capacity_matrix_sha256"], "capacity_matrix_sha256"
            ),
            maximum_capacity_numerator=_integer(
                obj["maximum_capacity_numerator"],
                "maximum_capacity_numerator",
                minimum=1,
            ),
            maximum_capacity_denominator=_integer(
                obj["maximum_capacity_denominator"],
                "maximum_capacity_denominator",
                minimum=1,
            ),
            overloaded_capacity_cells=_integer(
                obj["overloaded_capacity_cells"], "overloaded_capacity_cells"
            ),
        )


class FormulaEvidenceParserV2(Protocol):
    """Version-pinned parser for tensor inventory, executable, HLO and ops."""

    @property
    def identity(self) -> ContentRefV1: ...

    def parse_formula(
        self,
        *,
        run_contract: PrefixMFURunContractV2,
        formula: PrefixFlopFormulaV2,
        device_inventory: DeviceInventoryV1,
        artifacts: Mapping[str, ResolvedContentV2],
    ) -> MFUFormulaReconciliationReceiptPayloadV2: ...


class InputEvidenceParserV2(Protocol):
    """Version-pinned parser for packed manifests and segment ledgers."""

    @property
    def identity(self) -> ContentRefV1: ...

    def parse_input(
        self,
        *,
        run_contract: PrefixMFURunContractV2,
        input_slice: PackedInputSliceV2,
        parent_manifest: ResolvedContentV2,
        segment_ledger: ResolvedContentV2,
    ) -> MFUInputReductionReceiptPayloadV2: ...


class WindowEvidenceParserV2(Protocol):
    """Version-pinned parser for compile, trace, health and routing artifacts."""

    @property
    def identity(self) -> ContentRefV1: ...

    def parse_compile_warmup(
        self,
        *,
        run_contract: PrefixMFURunContractV2,
        device_inventory: DeviceInventoryV1,
        windows: tuple[PrefixMFUWindowEvidenceV2, ...],
        artifacts: Mapping[str, ResolvedContentV2],
    ) -> MFUCompileWarmupReceiptPayloadV2: ...

    def parse_window(
        self,
        *,
        run_contract: PrefixMFURunContractV2,
        entry: MFUWindowPlanEntryV2,
        input_slice: PackedInputSliceV2,
        evidence: PrefixMFUWindowEvidenceV2,
        device_inventory: DeviceInventoryV1,
        artifacts: Mapping[str, ResolvedContentV2],
    ) -> MFUWindowReductionReceiptPayloadV2: ...


_RECEIPT_MEDIA_BY_TYPE: dict[type[CanonicalRecord], str] = {
    MFUPredeclarationReceiptPayloadV2: MEDIA_PREDECLARATION_RECEIPT_V2,
    MFUCompileWarmupReceiptPayloadV2: MEDIA_WARMUP_RECEIPT_V2,
    MFUFormulaReconciliationReceiptPayloadV2: (MEDIA_FORMULA_RECONCILIATION_RECEIPT_V2),
    MFUInputReductionReceiptPayloadV2: MEDIA_INPUT_REDUCTION_RECEIPT_V2,
    MFUWindowReductionReceiptPayloadV2: MEDIA_WINDOW_REDUCTION_RECEIPT_V2,
}


class ControllerMFUEvidenceRegistryV2:
    """Publish controller-owned, immutable, generation-pinned V2 receipts.

    The registry does not bless parser outputs.  It only gives canonical
    receipt bytes a controller-owned name and exact generation.  The external
    production authority resolves and recomputes every receipt before use.
    """

    def __init__(
        self,
        store: JSONCASStore,
        *,
        policy: TrustedMFUAuthorityPolicyV2,
        campaign_id: str,
        campaign_head_uri: str,
        campaign_head_generation: int,
        campaign_head_revision: int,
        replica_region: str,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self._store = store
        self._policy = policy
        self._campaign_id = _identifier(campaign_id, "campaign_id")
        self._head_uri = require_string(campaign_head_uri, "campaign_head_uri")
        if not self._head_uri.startswith("gs://"):
            raise MFUAuthorityError("campaign_head_uri must be a gs:// URI")
        self._head_generation = _integer(
            campaign_head_generation, "campaign_head_generation", minimum=1
        )
        self._head_revision = _integer(campaign_head_revision, "campaign_head_revision")
        self._region = _identifier(replica_region, "replica_region")
        if self._region not in policy.allowed_replica_regions:
            raise MFUAuthorityError("registry region is not externally allowlisted")
        self._clock_ns = clock_ns or time.time_ns

    def _publish_record(
        self, record: CanonicalRecord, *, media_type: str
    ) -> ContentRefV1:
        value = record.canonical_bytes()
        uri = (
            f"{self._policy.controller_receipt_prefix}"
            f"{record.SCHEMA}/{record.digest}.json"
        )
        version = self._store.put_immutable(uri, record)
        if version.value != value or version.sha256 != record.digest:
            raise MFUAuthorityError("receipt store changed canonical bytes")
        crc = _crc32c_base64(value)
        replica = RegionalReplicaV1(
            region=self._region,
            uri=uri,
            generation=version.generation,
            size_bytes=len(value),
            crc32c=crc,
            sha256=record.digest,
        )
        return ContentRefV1(
            sha256=record.digest,
            size_bytes=len(value),
            media_type=media_type,
            replicas=(replica,),
        )

    def _publish_receipt(self, record: CanonicalRecord) -> ContentRefV1:
        media_type = _RECEIPT_MEDIA_BY_TYPE.get(type(record))
        if media_type is None:
            raise MFUAuthorityError("unsupported MFU receipt record type")
        return self._publish_record(record, media_type=media_type)

    def publish_parser_receipt(self, record: CanonicalRecord) -> ContentRefV1:
        """Publish a parser output; preregistration has a separate fenced API."""

        if isinstance(record, MFUPredeclarationReceiptPayloadV2):
            raise MFUAuthorityError(
                "predeclaration receipts must be minted by CampaignController"
            )
        return self._publish_receipt(record)

    def publish_verified_envelope(
        self, envelope: VerifiedMFUQualificationEnvelopeV2
    ) -> ContentRefV1:
        """Fail closed while no production sealing factory exists."""

        del envelope
        raise MFUAuthorityError(_PRODUCTION_DISABLED_MESSAGE)

    def _predeclare_from_controller(
        self,
        run_contract: PrefixMFURunContractV2,
        *,
        work_id: str,
        controller_fence: int,
        executor_id: str,
        lease_acquired_at_unix_ns: int,
        lease_expires_at_unix_ns: int,
        measurement_not_before_unix_ns: int,
    ) -> ContentRefV1:
        """Mint a preregistration after CampaignController lease admission."""

        if not isinstance(run_contract, PrefixMFURunContractV2):
            raise MFUAuthorityError("run_contract has the wrong record type")
        if not _same_payload(
            run_contract.semantic_authority, self._policy.authority_identity
        ):
            raise MFUAuthorityError(
                "run contract does not name the externally allowlisted authority"
            )
        registered = _integer(self._clock_ns(), "controller clock", minimum=1)
        payload = MFUPredeclarationReceiptPayloadV2(
            policy_sha256=self._policy.digest,
            authority_sha256=self._policy.authority_identity.sha256,
            campaign_id=self._campaign_id,
            campaign_head_uri=self._head_uri,
            campaign_head_generation=self._head_generation,
            campaign_head_revision=self._head_revision,
            work_id=_identifier(work_id, "work_id"),
            attempt_id=run_contract.attempt_id,
            executor_id=_identifier(executor_id, "executor_id"),
            scientific_identity_sha256=run_contract.scientific_identity.sha256,
            deployment_identity_sha256=run_contract.deployment_identity.sha256,
            run_contract_sha256=run_contract.digest,
            window_plan_sha256=run_contract.window_plan.digest,
            controller_fence=_integer(controller_fence, "controller_fence", minimum=1),
            lease_acquired_at_unix_ns=_integer(
                lease_acquired_at_unix_ns,
                "lease_acquired_at_unix_ns",
                minimum=1,
            ),
            lease_expires_at_unix_ns=_integer(
                lease_expires_at_unix_ns,
                "lease_expires_at_unix_ns",
                minimum=1,
            ),
            registered_at_unix_ns=registered,
            measurement_not_before_unix_ns=_integer(
                measurement_not_before_unix_ns,
                "measurement_not_before_unix_ns",
                minimum=1,
            ),
        )
        return self._publish_receipt(payload)


def _expected_formula_receipt(
    run: PrefixMFURunContractV2,
    formula: PrefixFlopFormulaV2,
) -> MFUFormulaReconciliationReceiptPayloadV2:
    coefficients_digest = canonical_sha256(
        {
            "schema": "moe-sae.r1.deepseek-r1-prefix-coefficients.v2",
            "coefficients": deepseek_r1_prefix_flop_coefficients(),
        }
    )
    return MFUFormulaReconciliationReceiptPayloadV2(
        run_contract_sha256=run.digest,
        formula_sha256=formula.digest,
        executable_sha256=run.executable.sha256,
        hlo_sha256=run.hlo.sha256,
        tensor_inventory_sha256=run.tensor_inventory.sha256,
        operation_inventory_sha256=formula.counted_operation_inventory.sha256,
        analytic_formula_sha256=formula.analytic_formula_sha256,
        coefficient_ledger_sha256=coefficients_digest,
        sequences_per_step=run.sequences_per_step,
        sequence_length=formula.sequence_length,
        counted_matmul_dtype=formula.counted_matmul_dtype,
        denominator_dtype=run.denominator_dtype,
    )


def _expected_input_receipt(
    run: PrefixMFURunContractV2,
    input_slice: PackedInputSliceV2,
) -> MFUInputReductionReceiptPayloadV2:
    return MFUInputReductionReceiptPayloadV2(
        run_contract_sha256=run.digest,
        input_slice_sha256=input_slice.digest,
        parent_input_manifest_sha256=input_slice.parent_input_manifest_sha256,
        segment_ledger_sha256=input_slice.segment_ledger.sha256,
        group_index=input_slice.group_index,
        data_kind=input_slice.data_kind,
        batch_start=input_slice.batch_start,
        batch_end=input_slice.batch_end,
        packed_sequence_count=input_slice.packed_sequence_count,
        histogram_sha256=input_slice.histogram.digest,
        valid_tokens=input_slice.histogram.valid_tokens,
        causal_pairs=input_slice.histogram.causal_pairs,
    )


def _window_artifact_refs(
    evidence: PrefixMFUWindowEvidenceV2,
) -> tuple[ContentRefV1, ...]:
    references = (
        evidence.step_trace,
        evidence.timing_blocks,
        evidence.profiler_trace,
        evidence.host_metrics,
        evidence.integrity_events,
        evidence.routing_capacity_ledger,
    )
    if evidence.persistence_receipt is not None:
        return (*references, evidence.persistence_receipt)
    return references


class ProtocolOracleMFUEvidenceAuthorityV2:
    """Generation-pinned semantic oracle for protocol and parser testing.

    Parser identity is still supplied by the parser object, so this class is
    deliberately *not* a production trust boundary.  It may prove that one
    parser implementation agrees with the canonical reducer and receipts, but
    its output remains UNVERIFIED and it cannot mint or publish headline PASS.
    """

    def __init__(
        self,
        *,
        policy: TrustedMFUAuthorityPolicyV2,
        resolver: GenerationPinnedContentResolverV2,
        formula_parser: FormulaEvidenceParserV2,
        input_parser: InputEvidenceParserV2,
        window_parser: WindowEvidenceParserV2,
    ) -> None:
        if not isinstance(policy, TrustedMFUAuthorityPolicyV2):
            raise MFUAuthorityError("policy must be externally provisioned V2 policy")
        identities = (
            (getattr(resolver, "identity", None), policy.resolver_identity, "resolver"),
            (
                getattr(formula_parser, "identity", None),
                policy.formula_parser_identity,
                "formula parser",
            ),
            (
                getattr(input_parser, "identity", None),
                policy.input_parser_identity,
                "input parser",
            ),
            (
                getattr(window_parser, "identity", None),
                policy.window_parser_identity,
                "window parser",
            ),
        )
        for actual, expected, name in identities:
            if (
                not isinstance(actual, ContentRefV1)
                or actual.canonical_bytes() != expected.canonical_bytes()
            ):
                raise MFUAuthorityError(f"{name} is not externally allowlisted")
        for reference in (
            policy.authority_identity,
            policy.resolver_identity,
            policy.formula_parser_identity,
            policy.input_parser_identity,
            policy.window_parser_identity,
        ):
            try:
                resolved_identity = resolver.resolve(reference)
            except Exception as error:
                raise MFUAuthorityError(
                    "an allowlisted authority component identity did not resolve"
                ) from error
            if resolved_identity.replica.region not in policy.allowed_replica_regions:
                raise MFUAuthorityError(
                    "an authority component identity resolved outside allowed regions"
                )
        self._policy = policy
        self._resolver = resolver
        self._formula_parser = formula_parser
        self._input_parser = input_parser
        self._window_parser = window_parser
        self._resolved: dict[str, ResolvedContentV2] = {}

    @property
    def policy(self) -> TrustedMFUAuthorityPolicyV2:
        return self._policy

    @property
    def identity(self) -> ContentRefV1:
        return self._policy.authority_identity

    def _resolve(self, reference: ContentRefV1) -> ResolvedContentV2:
        key = reference.digest
        cached = self._resolved.get(key)
        if cached is not None:
            return cached
        try:
            result = self._resolver.resolve(reference)
        except Exception as error:
            raise MFUAuthorityError(
                "generation-pinned content resolution failed"
            ) from error
        if result.replica.region not in self._policy.allowed_replica_regions:
            raise MFUAuthorityError("resolved replica region is not allowlisted")
        # Receipt/control objects are reused repeatedly and are small.  Raw
        # Xprof and routing payloads can be many GiB, so retaining every
        # resolved byte buffer across a 12-window qualification is forbidden.
        if reference.size_bytes <= _MAX_IN_MEMORY_RESOLUTION_CACHE_BYTES:
            self._resolved[key] = result
        return result

    def verify_content(self, reference: ContentRefV1) -> bool:
        self._resolve(reference)
        return True

    def resolve_content(self, reference: ContentRefV1) -> ResolvedContentV2:
        """Resolve one exact-generation object for protocol-oracle inspection."""

        return self._resolve(reference)

    def _load_receipt(
        self,
        reference: ContentRefV1,
        record_type: type[CanonicalRecord],
        media_type: str,
    ) -> CanonicalRecord:
        _content(reference, "receipt", media_type)
        if any(
            not replica.uri.startswith(self._policy.controller_receipt_prefix)
            for replica in reference.replicas
        ):
            raise MFUAuthorityError("receipt is outside the trusted controller prefix")
        resolved = self._resolve(reference)
        try:
            parsed = parse_json(resolved.value)
            if not isinstance(parsed, Mapping):
                raise MFUAuthorityError("receipt JSON must be an object")
            result = record_type.from_dict(parsed)
        except Exception as error:
            if isinstance(error, MFUAuthorityError):
                raise
            raise MFUAuthorityError("receipt payload failed strict parsing") from error
        if result.canonical_bytes() != resolved.value:
            raise MFUAuthorityError("receipt bytes are not canonical JSON")
        return result

    def _artifacts(
        self, pairs: Sequence[tuple[str, ContentRefV1]]
    ) -> Mapping[str, ResolvedContentV2]:
        result: dict[str, ResolvedContentV2] = {}
        for name, reference in pairs:
            if name in result:
                raise MFUAuthorityError("duplicate parser artifact role")
            result[name] = self._resolve(reference)
        return result

    def verify_semantics(
        self,
        *,
        run_contract: PrefixMFURunContractV2,
        formula: PrefixFlopFormulaV2,
        device_inventory: DeviceInventoryV1,
        windows: tuple[PrefixMFUWindowEvidenceV2, ...],
        receipts: MFUSemanticReceiptsV2,
    ) -> bool:
        if not _same_payload(run_contract.semantic_authority, self.identity):
            raise MFUAuthorityError("run contract names a non-allowlisted authority")
        if not _same_payload(receipts.authority_identity, self.identity):
            raise MFUAuthorityError("receipts name a non-allowlisted authority")
        run_contract.verify_dependencies(formula, device_inventory)

        predecl = self._load_receipt(
            receipts.predeclaration_receipt,
            MFUPredeclarationReceiptPayloadV2,
            MEDIA_PREDECLARATION_RECEIPT_V2,
        )
        assert isinstance(predecl, MFUPredeclarationReceiptPayloadV2)
        if (
            predecl.policy_sha256 != self._policy.digest
            or predecl.authority_sha256 != self.identity.sha256
            or predecl.attempt_id != run_contract.attempt_id
            or predecl.scientific_identity_sha256
            != run_contract.scientific_identity.sha256
            or predecl.deployment_identity_sha256
            != run_contract.deployment_identity.sha256
            or predecl.run_contract_sha256 != run_contract.digest
            or predecl.window_plan_sha256 != run_contract.window_plan.digest
        ):
            raise MFUAuthorityError(
                "predeclaration is not bound to this run and policy"
            )

        formula_receipt = self._load_receipt(
            receipts.formula_reconciliation_receipt,
            MFUFormulaReconciliationReceiptPayloadV2,
            MEDIA_FORMULA_RECONCILIATION_RECEIPT_V2,
        )
        assert isinstance(formula_receipt, MFUFormulaReconciliationReceiptPayloadV2)
        formula_artifacts = self._artifacts(
            (
                ("tensor_inventory", run_contract.tensor_inventory),
                ("executable", run_contract.executable),
                ("hlo", run_contract.hlo),
                ("operation_inventory", formula.counted_operation_inventory),
            )
        )
        parsed_formula = self._formula_parser.parse_formula(
            run_contract=run_contract,
            formula=formula,
            device_inventory=device_inventory,
            artifacts=formula_artifacts,
        )
        expected_formula = _expected_formula_receipt(run_contract, formula)
        if not isinstance(parsed_formula, MFUFormulaReconciliationReceiptPayloadV2):
            raise MFUAuthorityError("formula parser returned the wrong record type")
        if not (
            parsed_formula.canonical_bytes()
            == formula_receipt.canonical_bytes()
            == expected_formula.canonical_bytes()
        ):
            raise MFUAuthorityError("formula receipt does not match parsed HLO/formula")

        compile_artifacts: list[tuple[str, ContentRefV1]] = [
            ("executable", run_contract.executable),
            ("hlo", run_contract.hlo),
            ("device_inventory", run_contract.device_inventory),
        ]
        for index, window in enumerate(windows):
            compile_artifacts.append((f"window_{index}_step_trace", window.step_trace))
        compile_receipt = self._load_receipt(
            receipts.compile_warmup_receipt,
            MFUCompileWarmupReceiptPayloadV2,
            MEDIA_WARMUP_RECEIPT_V2,
        )
        assert isinstance(compile_receipt, MFUCompileWarmupReceiptPayloadV2)
        parsed_compile = self._window_parser.parse_compile_warmup(
            run_contract=run_contract,
            device_inventory=device_inventory,
            windows=windows,
            artifacts=self._artifacts(compile_artifacts),
        )
        if not isinstance(parsed_compile, MFUCompileWarmupReceiptPayloadV2):
            raise MFUAuthorityError(
                "window parser returned the wrong warmup record type"
            )
        if parsed_compile.canonical_bytes() != compile_receipt.canonical_bytes():
            raise MFUAuthorityError("compile/warmup receipt does not match raw traces")
        first_entry = run_contract.window_plan.entries[0]
        if (
            compile_receipt.run_contract_sha256 != run_contract.digest
            or compile_receipt.attempt_id != run_contract.attempt_id
            or compile_receipt.executable_sha256 != run_contract.executable.sha256
            or compile_receipt.hlo_sha256 != run_contract.hlo.sha256
            or compile_receipt.device_inventory_sha256
            != run_contract.device_inventory.sha256
            or compile_receipt.controller_fence != predecl.controller_fence
            or compile_receipt.compile_started_at_unix_ns
            < predecl.lease_acquired_at_unix_ns
            or compile_receipt.completed_warmup_steps < run_contract.warmup_steps
            or compile_receipt.first_measurement_step != first_entry.step_start
            or compile_receipt.measurement_recompilations
            != sum(window.recompilations for window in windows)
        ):
            raise MFUAuthorityError("compile/warmup receipt is not bound to the run")

        input_receipts = tuple(
            self._load_receipt(
                reference,
                MFUInputReductionReceiptPayloadV2,
                MEDIA_INPUT_REDUCTION_RECEIPT_V2,
            )
            for reference in receipts.input_reduction_receipts
        )
        for input_slice, stored in zip(
            run_contract.window_plan.input_slices, input_receipts, strict=True
        ):
            assert isinstance(stored, MFUInputReductionReceiptPayloadV2)
            parent = (
                run_contract.real_input_manifest
                if input_slice.data_kind == "REAL"
                else run_contract.synthetic_input_manifest
            )
            parsed = self._input_parser.parse_input(
                run_contract=run_contract,
                input_slice=input_slice,
                parent_manifest=self._resolve(parent),
                segment_ledger=self._resolve(input_slice.segment_ledger),
            )
            expected = _expected_input_receipt(run_contract, input_slice)
            if not isinstance(parsed, MFUInputReductionReceiptPayloadV2):
                raise MFUAuthorityError("input parser returned the wrong record type")
            if not (
                parsed.canonical_bytes()
                == stored.canonical_bytes()
                == expected.canonical_bytes()
            ):
                raise MFUAuthorityError("input reduction does not match raw input data")

        stored_windows = tuple(
            self._load_receipt(
                reference,
                MFUWindowReductionReceiptPayloadV2,
                MEDIA_WINDOW_REDUCTION_RECEIPT_V2,
            )
            for reference in receipts.window_reduction_receipts
        )
        previous_finished = 0
        for entry, evidence, stored in zip(
            run_contract.window_plan.entries, windows, stored_windows, strict=True
        ):
            assert isinstance(stored, MFUWindowReductionReceiptPayloadV2)
            artifact_refs = _window_artifact_refs(evidence)
            parsed = self._window_parser.parse_window(
                run_contract=run_contract,
                entry=entry,
                input_slice=run_contract.window_plan.input_slice(
                    entry.input_slice_sha256
                ),
                evidence=evidence,
                device_inventory=device_inventory,
                artifacts=self._artifacts(
                    tuple(
                        (f"artifact_{index}", reference)
                        for index, reference in enumerate(artifact_refs)
                    )
                ),
            )
            if not isinstance(parsed, MFUWindowReductionReceiptPayloadV2):
                raise MFUAuthorityError("window parser returned the wrong record type")
            if parsed.canonical_bytes() != stored.canonical_bytes():
                raise MFUAuthorityError("window receipt does not match raw artifacts")
            expected_cells = (
                (entry.step_end - entry.step_start)
                * (DEEPSEEK_R1_PREFIX_GEOMETRY_V1.completed_moe_blocks + 1)
                * DEEPSEEK_R1_PREFIX_GEOMETRY_V1.routed_experts
            )
            if (
                stored.run_contract_sha256 != run_contract.digest
                or stored.attempt_id != run_contract.attempt_id
                or stored.measurement_execution_id
                != compile_receipt.measurement_execution_id
                or stored.window_id != entry.window_id
                or stored.input_slice_sha256 != entry.input_slice_sha256
                or stored.executable_sha256 != run_contract.executable.sha256
                or stored.hlo_sha256 != run_contract.hlo.sha256
                or stored.device_inventory_sha256
                != run_contract.device_inventory.sha256
                or stored.controller_fence != predecl.controller_fence
                or stored.device_ids_sha256 != _device_ids_digest(device_inventory)
                or stored.artifact_bundle_sha256 != _receipt_ref_digest(artifact_refs)
                or stored.observation_sha256 != window_observation_digest_v2(evidence)
                or stored.step_start != entry.step_start
                or stored.step_end != entry.step_end
                or stored.measurement_finished_at_unix_ns
                - stored.measurement_started_at_unix_ns
                != evidence.wall_time_ns
                or stored.capacity_steps != entry.step_end - entry.step_start
                or stored.capacity_router_invocations
                != DEEPSEEK_R1_PREFIX_GEOMETRY_V1.completed_moe_blocks + 1
                or stored.capacity_routed_experts
                != DEEPSEEK_R1_PREFIX_GEOMETRY_V1.routed_experts
                or stored.capacity_cell_count != expected_cells
                or stored.capacity_matrix_sha256
                != evidence.routing_capacity_ledger.sha256
                or stored.overloaded_capacity_cells != 0
            ):
                raise MFUAuthorityError(
                    "window receipt is not exactly bound to evidence"
                )
            if stored.measurement_started_at_unix_ns < previous_finished:
                raise MFUAuthorityError(
                    "measurement windows overlap in wall-clock time"
                )
            if (
                stored.measurement_started_at_unix_ns
                < predecl.measurement_not_before_unix_ns
            ):
                raise MFUAuthorityError(
                    "measurement began before controller preregistration"
                )
            if (
                stored.measurement_finished_at_unix_ns
                > predecl.lease_expires_at_unix_ns
            ):
                raise MFUAuthorityError(
                    "measurement continued beyond its admitted lease"
                )
            previous_finished = stored.measurement_finished_at_unix_ns

        if (
            compile_receipt.first_measurement_started_at_unix_ns
            != stored_windows[0].measurement_started_at_unix_ns
            or compile_receipt.warmup_finished_at_unix_ns
            > stored_windows[0].measurement_started_at_unix_ns
            or predecl.registered_at_unix_ns
            > stored_windows[0].measurement_started_at_unix_ns
        ):
            raise MFUAuthorityError(
                "preregistration/warmup did not precede measurement"
            )
        return True

    def qualify_unverified(
        self,
        *,
        run_contract: PrefixMFURunContractV2,
        formula: PrefixFlopFormulaV2,
        device_inventory: DeviceInventoryV1,
        windows: Sequence[PrefixMFUWindowEvidenceV2],
        semantic_receipts: MFUSemanticReceiptsV2,
    ) -> PrefixMFUQualificationV2:
        """Run semantic checks and return only an UNVERIFIED qualification."""

        return qualify_prefix_mfu_v2(
            run_contract=run_contract,
            formula=formula,
            device_inventory=device_inventory,
            windows=windows,
            semantic_receipts=semantic_receipts,
            evidence_authority=self,
        )

    def qualify_and_seal(self, **_evidence: Any) -> VerifiedMFUQualificationEnvelopeV2:
        """Reject headline sealing from an injectable protocol oracle."""

        raise MFUAuthorityError(_PRODUCTION_DISABLED_MESSAGE)

    def latest_measurement_finished_at_unix_ns(
        self, receipts: MFUSemanticReceiptsV2
    ) -> int:
        if not isinstance(receipts, MFUSemanticReceiptsV2):
            raise MFUAuthorityError("semantic receipts have the wrong record type")
        latest = self._load_receipt(
            receipts.window_reduction_receipts[-1],
            MFUWindowReductionReceiptPayloadV2,
            MEDIA_WINDOW_REDUCTION_RECEIPT_V2,
        )
        assert isinstance(latest, MFUWindowReductionReceiptPayloadV2)
        return latest.measurement_finished_at_unix_ns


class ProductionMFUEvidenceAuthorityV2:
    """Reserved production authority; construction is intentionally disabled.

    The current repository has parser protocols and a useful protocol oracle,
    but no closed mapping from externally pinned identities to exact real
    parser implementations and no independently replayed controller-admission
    event.  Accepting injectable parser objects here would turn a self-reported
    identity into headline authority, so every public construction fails.
    """

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise MFUAuthorityError(_PRODUCTION_DISABLED_MESSAGE)


@dataclass(frozen=True, slots=True, init=False)
class VerifiedMFUQualificationEnvelopeV2(CanonicalRecord):
    """Reserved PASS envelope; no instance is admissible in this scaffold."""

    SCHEMA: ClassVar[str] = "moe-sae.r1.verified-mfu-qualification.v2"

    qualification: PrefixMFUQualificationV2
    policy_sha256: str
    authority_identity: ContentRefV1
    semantic_receipts_sha256: str
    verified_at_unix_ns: int

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise MFUAuthorityError(_PRODUCTION_DISABLED_MESSAGE)

    @classmethod
    def _from_verified(
        cls,
        *,
        qualification: PrefixMFUQualificationV2,
        policy: TrustedMFUAuthorityPolicyV2,
        authority_identity: ContentRefV1,
        verified_at_unix_ns: int,
    ) -> Self:
        del qualification, policy, authority_identity, verified_at_unix_ns
        raise MFUAuthorityError(_PRODUCTION_DISABLED_MESSAGE)

    @property
    def qualification_status(self) -> str:
        return "PASS"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "qualification": self.qualification.as_dict(),
            "policy_sha256": self.policy_sha256,
            "authority_identity": self.authority_identity.as_dict(),
            "semantic_receipts_sha256": self.semantic_receipts_sha256,
            "verified_at_unix_ns": self.verified_at_unix_ns,
            "qualification_status": self.qualification_status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        del value
        raise MFUAuthorityError(_PRODUCTION_DISABLED_MESSAGE)


class MFUReportConsumerV2:
    """Reserved report boundary; disabled until the closed factory exists."""

    def __init__(
        self,
        *,
        policy: TrustedMFUAuthorityPolicyV2,
        authority: ProductionMFUEvidenceAuthorityV2,
    ) -> None:
        del policy, authority
        raise MFUAuthorityError(_PRODUCTION_DISABLED_MESSAGE)

    def load_verified(
        self, value: str | bytes | Mapping[str, Any]
    ) -> VerifiedMFUQualificationEnvelopeV2:
        if isinstance(value, Mapping):
            parsed: Any = value
        else:
            parsed = parse_json(value)
        obj = require_record_object(
            parsed,
            schema=VerifiedMFUQualificationEnvelopeV2.SCHEMA,
            fields=(
                "qualification",
                "policy_sha256",
                "authority_identity",
                "semantic_receipts_sha256",
                "verified_at_unix_ns",
                "qualification_status",
            ),
        )
        if (
            require_string(obj["qualification_status"], "qualification_status")
            != "PASS"
        ):
            raise MFUAuthorityError("verified envelope must declare PASS")
        if _sha(obj["policy_sha256"], "policy_sha256") != self._policy.digest:
            raise MFUAuthorityError("verified envelope uses a non-allowlisted policy")
        authority_identity = ContentRefV1.from_dict(obj["authority_identity"])
        if (
            authority_identity.canonical_bytes()
            != self._policy.authority_identity.canonical_bytes()
        ):
            raise MFUAuthorityError(
                "verified envelope uses a non-allowlisted authority"
            )
        qualification_value = obj["qualification"]
        if not isinstance(qualification_value, Mapping):
            raise MFUAuthorityError("qualification must be an object")
        qualification = PrefixMFUQualificationV2.from_dict_verified(
            qualification_value,
            evidence_authority=self._authority,
        )
        receipts_sha = _sha(obj["semantic_receipts_sha256"], "semantic_receipts_sha256")
        if receipts_sha != qualification.semantic_receipts.digest:
            raise MFUAuthorityError("envelope splices different semantic receipts")
        verified_at = _integer(
            obj["verified_at_unix_ns"], "verified_at_unix_ns", minimum=1
        )
        if verified_at < self._authority.latest_measurement_finished_at_unix_ns(
            qualification.semantic_receipts
        ):
            raise MFUAuthorityError("envelope verification predates measurement")
        result = VerifiedMFUQualificationEnvelopeV2._from_verified(
            qualification=qualification,
            policy=self._policy,
            authority_identity=authority_identity,
            verified_at_unix_ns=verified_at,
        )
        if result.as_dict() != obj:
            raise MFUAuthorityError("verified envelope is not canonical")
        return result

    def load_verified_ref(
        self, reference: ContentRefV1
    ) -> VerifiedMFUQualificationEnvelopeV2:
        """Resolve and reverify a generation-pinned published PASS envelope."""

        _content(
            reference,
            "verified_envelope",
            MEDIA_VERIFIED_QUALIFICATION_ENVELOPE_V2,
        )
        if any(
            not replica.uri.startswith(self._policy.controller_receipt_prefix)
            for replica in reference.replicas
        ):
            raise MFUAuthorityError(
                "verified envelope is outside the trusted controller prefix"
            )
        resolved = self._authority.resolve_content(reference)
        return self.load_verified(resolved.value)


__all__ = [
    "FormulaEvidenceParserV2",
    "GCSGenerationPinnedContentResolverV2",
    "GenerationPinnedContentResolverV2",
    "InputEvidenceParserV2",
    "JSONCASGenerationPinnedResolverV2",
    "MEDIA_AUTHORITY_POLICY_V2",
    "MEDIA_CONTENT_RESOLVER_IDENTITY_V2",
    "MEDIA_FORMULA_PARSER_IDENTITY_V2",
    "MEDIA_INPUT_PARSER_IDENTITY_V2",
    "MEDIA_VERIFIED_QUALIFICATION_ENVELOPE_V2",
    "MEDIA_WINDOW_PARSER_IDENTITY_V2",
    "MFUAuthorityError",
    "MFUCompileWarmupReceiptPayloadV2",
    "MFUFormulaReconciliationReceiptPayloadV2",
    "MFUInputReductionReceiptPayloadV2",
    "MFUPredeclarationReceiptPayloadV2",
    "MFUReportConsumerV2",
    "MFUWindowReductionReceiptPayloadV2",
    "ProtocolOracleMFUEvidenceAuthorityV2",
    "ProductionMFUEvidenceAuthorityV2",
    "ResolvedContentV2",
    "TrustedMFUAuthorityPolicyV2",
    "VerifiedMFUQualificationEnvelopeV2",
    "WindowEvidenceParserV2",
    "window_observation_digest_v2",
]
