"""Compare-and-swap storage primitives for the R1 controller.

Bulk artifacts use the repository's immutable object store.  This deliberately
separate interface is for the small mutable JSON documents that grant leases,
advance fences, and publish campaign state.  GCS can implement the same API
with object-generation preconditions; the in-memory implementation below is a
deterministic semantic oracle for controller tests.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import re
import threading
from typing import Any, Callable, Mapping, Protocol

from .protocol import (
    CanonicalRecord,
    ProtocolError,
    canonical_json_bytes,
    parse_json,
)


_SHA256 = re.compile(r"[0-9a-f]{64}")
_SCHEMA = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\.v[1-9][0-9]*")


class ControlStoreError(RuntimeError):
    """Base class for control-store failures."""


class CASConflict(ControlStoreError):
    """A create or generation-matched mutation lost a race."""

    def __init__(
        self,
        uri: str,
        *,
        expected_generation: int | None,
        actual_generation: int | None,
    ) -> None:
        self.uri = uri
        self.expected_generation = expected_generation
        self.actual_generation = actual_generation
        super().__init__(
            f"CAS conflict for {uri!r}: expected generation "
            f"{expected_generation!r}, actual {actual_generation!r}"
        )


class ImmutableConflict(ControlStoreError):
    """An immutable name already exists with different content."""


@dataclass(frozen=True, slots=True)
class VersionedJSON:
    """Canonical JSON bytes pinned to one store generation."""

    uri: str
    value: bytes
    generation: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_uri(self.uri)
        if not isinstance(self.value, bytes):
            raise TypeError("value must be canonical JSON bytes")
        if type(self.generation) is not int or self.generation <= 0:
            raise ValueError("generation must be a positive integer")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        if hashlib.sha256(self.value).hexdigest() != self.sha256:
            raise ValueError("sha256 does not match value")
        if _canonicalize_value(self.value) != self.value:
            raise ValueError("value is not canonical JSON")

    def decoded(self) -> Mapping[str, Any]:
        parsed = parse_json(self.value)
        if not isinstance(parsed, Mapping):  # defended by _canonicalize_value
            raise AssertionError("stored versioned JSON is not an object")
        return parsed


class JSONCASStore(Protocol):
    """Minimal generation-precondition API used by the controller."""

    def read(self, uri: str) -> VersionedJSON | None: ...

    def create(self, uri: str, value: Any) -> VersionedJSON:
        """Create a mutable document iff the name is absent."""

    def compare_and_swap(
        self,
        uri: str,
        expected_generation: int,
        value: Any,
    ) -> VersionedJSON:
        """Replace a mutable document iff its generation still matches."""

    def put_immutable(self, uri: str, value: Any) -> VersionedJSON:
        """Publish once; an identical retry is idempotent."""


def _validate_uri(uri: str) -> None:
    if not isinstance(uri, str) or not uri or uri != uri.strip():
        raise ValueError("uri must be a non-empty string without outer whitespace")
    if any(ord(character) < 0x20 for character in uri):
        raise ValueError("uri must not contain control characters")


def _canonicalize_value(value: Any) -> bytes:
    if isinstance(value, CanonicalRecord):
        obj: Any = value.as_dict()
    elif isinstance(value, (str, bytes)):
        obj = parse_json(value)
    else:
        obj = value
    if not isinstance(obj, Mapping):
        raise ProtocolError("control-store JSON must be an object")
    schema = obj.get("schema")
    if not isinstance(schema, str) or _SCHEMA.fullmatch(schema) is None:
        raise ProtocolError(
            "control-store JSON requires a versioned schema ending in .v<N>"
        )
    return canonical_json_bytes(obj)


class MemoryJSONCASStore:
    """Thread-safe, deterministic implementation of :class:`JSONCASStore`.

    ``generation_factory`` is called under the store lock only after an
    operation has won its precondition.  Injecting an iterator's ``__next__``
    makes generation behavior exact in tests.  Returned generations must be
    positive and strictly increasing across this store instance, matching the
    only property the controller is allowed to rely upon.
    """

    def __init__(
        self,
        *,
        generation_factory: Callable[[], int] | None = None,
    ) -> None:
        counter = itertools.count(1)
        self._generation_factory = generation_factory or counter.__next__
        self._last_generation = 0
        self._values: dict[str, VersionedJSON] = {}
        self._immutable: set[str] = set()
        self._lock = threading.RLock()

    def _next_generation(self) -> int:
        generation = self._generation_factory()
        if type(generation) is not int or generation <= self._last_generation:
            raise ControlStoreError(
                "generation_factory must return strictly increasing positive integers"
            )
        self._last_generation = generation
        return generation

    @staticmethod
    def _version(uri: str, value: bytes, generation: int) -> VersionedJSON:
        return VersionedJSON(
            uri=uri,
            value=value,
            generation=generation,
            sha256=hashlib.sha256(value).hexdigest(),
        )

    def read(self, uri: str) -> VersionedJSON | None:
        _validate_uri(uri)
        with self._lock:
            return self._values.get(uri)

    def create(self, uri: str, value: Any) -> VersionedJSON:
        _validate_uri(uri)
        encoded = _canonicalize_value(value)
        with self._lock:
            current = self._values.get(uri)
            if current is not None:
                raise CASConflict(
                    uri,
                    expected_generation=None,
                    actual_generation=current.generation,
                )
            version = self._version(uri, encoded, self._next_generation())
            self._values[uri] = version
            return version

    def compare_and_swap(
        self,
        uri: str,
        expected_generation: int,
        value: Any,
    ) -> VersionedJSON:
        _validate_uri(uri)
        if type(expected_generation) is not int or expected_generation <= 0:
            raise ValueError("expected_generation must be a positive integer")
        encoded = _canonicalize_value(value)
        with self._lock:
            current = self._values.get(uri)
            actual = None if current is None else current.generation
            if current is None or current.generation != expected_generation:
                raise CASConflict(
                    uri,
                    expected_generation=expected_generation,
                    actual_generation=actual,
                )
            if uri in self._immutable:
                raise ImmutableConflict(f"immutable JSON cannot be replaced: {uri}")
            version = self._version(uri, encoded, self._next_generation())
            self._values[uri] = version
            return version

    def put_immutable(self, uri: str, value: Any) -> VersionedJSON:
        _validate_uri(uri)
        encoded = _canonicalize_value(value)
        with self._lock:
            current = self._values.get(uri)
            if current is not None:
                if uri not in self._immutable:
                    raise ImmutableConflict(
                        f"name already belongs to a mutable JSON document: {uri}"
                    )
                if current.value != encoded:
                    raise ImmutableConflict(
                        f"immutable JSON exists with different content: {uri}"
                    )
                return current
            version = self._version(uri, encoded, self._next_generation())
            self._values[uri] = version
            self._immutable.add(uri)
            return version


__all__ = [
    "CASConflict",
    "ControlStoreError",
    "ImmutableConflict",
    "JSONCASStore",
    "MemoryJSONCASStore",
    "VersionedJSON",
]
