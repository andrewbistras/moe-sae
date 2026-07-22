"""Fixed, non-interactive process boundary for an R1 TPU worker image.

This module intentionally implements only the local launch boundary in the
current tranche.  It validates the baked process contract, then fails closed
before metadata, JAX, checkpoint, or controller access because controller-side
placement proof and 16-host runtime admission do not exist yet.  An image that
contains this module is therefore buildable but cannot be called qualified.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import sys
from typing import Sequence


ENTRYPOINT_CONTRACT = "moe-sae-r1-fixed-systemd-entrypoint-v1"
ENTRYPOINT_STATE = "STATIC_CONTRACT_ONLY"
EXPECTED_METADATA_KEY = "moe-sae-r1-launch-ref"
EXPECTED_RUNTIME_USER = "moe_sae"
EXPECTED_UID = 20001
_METADATA_KEY = re.compile(r"[a-z][a-z0-9-]{0,62}")
_FORBIDDEN_CREDENTIAL_ENV = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_OAUTH_ACCESS_TOKEN",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "SSH_AUTH_SOCK",
    }
)


class RuntimeEntrypointError(RuntimeError):
    """The immutable local launch boundary failed closed."""


@dataclass(frozen=True, slots=True)
class LocalLaunchBoundaryV1:
    metadata_key: str
    uid: int
    user: str
    stdin_is_tty: bool
    stdout_is_tty: bool
    stderr_is_tty: bool

    def validate(self, environment: dict[str, str]) -> None:
        if self.metadata_key != EXPECTED_METADATA_KEY:
            raise RuntimeEntrypointError("metadata key is not the baked launch key")
        if _METADATA_KEY.fullmatch(self.metadata_key) is None:
            raise RuntimeEntrypointError("metadata key is malformed")
        if self.uid != EXPECTED_UID or self.user != EXPECTED_RUNTIME_USER:
            raise RuntimeEntrypointError("worker is not the baked unprivileged user")
        if self.stdin_is_tty or self.stdout_is_tty or self.stderr_is_tty:
            raise RuntimeEntrypointError("worker lifetime must not be interactive")
        present = sorted(_FORBIDDEN_CREDENTIAL_ENV.intersection(environment))
        if present:
            raise RuntimeEntrypointError(
                "human or file-backed credential environment is forbidden: "
                + ", ".join(present)
            )
        expected = {
            "HOME": "/var/lib/moe-sae-r1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": "/var/tmp/moe-sae-r1",
            "XDG_CACHE_HOME": "/var/cache/moe-sae-r1",
        }
        for name, value in expected.items():
            if environment.get(name) != value:
                raise RuntimeEntrypointError(f"baked environment mismatch for {name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--metadata-key", required=True)
    return parser.parse_args(argv)


def observe_local_boundary(metadata_key: str) -> LocalLaunchBoundaryV1:
    try:
        user = Path("/proc/self/loginuid").read_text(encoding="ascii").strip()
    except OSError:
        user = ""
    # loginuid is intentionally not used as identity: system services commonly
    # report an unset audit login. The fixed uid is mapped without shelling out.
    del user
    import pwd

    uid = os.getuid()
    return LocalLaunchBoundaryV1(
        metadata_key=metadata_key,
        uid=uid,
        user=pwd.getpwuid(uid).pw_name,
        stdin_is_tty=sys.stdin.isatty(),
        stdout_is_tty=sys.stdout.isatty(),
        stderr_is_tty=sys.stderr.isatty(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    boundary = observe_local_boundary(args.metadata_key)
    boundary.validate(dict(os.environ))
    raise RuntimeEntrypointError(
        "production admission is unavailable: controller-side Compute placement "
        "proof and atomic 16-host JAX admission have not been implemented"
    )


if __name__ == "__main__":  # pragma: no cover - exercised through main
    main()
