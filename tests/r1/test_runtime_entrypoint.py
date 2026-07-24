from __future__ import annotations

import pytest

from moe_sae.r1.runtime_entrypoint import (
    EXPECTED_METADATA_KEY,
    EXPECTED_RUNTIME_USER,
    EXPECTED_UID,
    LocalLaunchBoundaryV1,
    RuntimeEntrypointError,
    main,
)


ENVIRONMENT = {
    "HOME": "/var/lib/moe-sae-r1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONUNBUFFERED": "1",
    "TMPDIR": "/run/moe-sae-r1",
    "XDG_CACHE_HOME": "/var/cache/moe-sae-r1",
}


def _boundary(**changes: object) -> LocalLaunchBoundaryV1:
    values = {
        "metadata_key": EXPECTED_METADATA_KEY,
        "uid": EXPECTED_UID,
        "user": EXPECTED_RUNTIME_USER,
        "stdin_is_tty": False,
        "stdout_is_tty": False,
        "stderr_is_tty": False,
    }
    values.update(changes)
    return LocalLaunchBoundaryV1(**values)  # type: ignore[arg-type]


def test_local_boundary_accepts_only_baked_noninteractive_shape() -> None:
    _boundary().validate(dict(ENVIRONMENT))

    with pytest.raises(RuntimeEntrypointError, match="unprivileged user"):
        _boundary(uid=0, user="root").validate(dict(ENVIRONMENT))
    with pytest.raises(RuntimeEntrypointError, match="interactive"):
        _boundary(stdout_is_tty=True).validate(dict(ENVIRONMENT))
    with pytest.raises(RuntimeEntrypointError, match="credential"):
        _boundary().validate({**ENVIRONMENT, "GOOGLE_APPLICATION_CREDENTIALS": "/x"})
    with pytest.raises(RuntimeEntrypointError, match="baked environment mismatch"):
        _boundary().validate({**ENVIRONMENT, "TMPDIR": "/tmp"})


def test_entrypoint_fails_before_production_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "moe_sae.r1.runtime_entrypoint.observe_local_boundary",
        lambda metadata_key: _boundary(metadata_key=metadata_key),
    )
    monkeypatch.setattr("moe_sae.r1.runtime_entrypoint.os.environ", ENVIRONMENT)
    with pytest.raises(
        RuntimeEntrypointError, match="production admission is unavailable"
    ):
        main(["--metadata-key", EXPECTED_METADATA_KEY])


def test_entrypoint_rejects_arbitrary_metadata_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "moe_sae.r1.runtime_entrypoint.observe_local_boundary",
        lambda metadata_key: _boundary(metadata_key=metadata_key),
    )
    monkeypatch.setattr("moe_sae.r1.runtime_entrypoint.os.environ", ENVIRONMENT)
    with pytest.raises(RuntimeEntrypointError, match="baked launch key"):
        main(["--metadata-key", "operator-command"])
