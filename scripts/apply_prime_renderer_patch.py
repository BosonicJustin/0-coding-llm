#!/usr/bin/env python3
"""Install the frozen StarCoder2 renderer into exact Prime source pins.

The operation is fail-closed and idempotent.  It refuses a different commit,
an initially dirty renderer checkout, or conflicting destination modules.  Use
``--check-only`` in image builds before allowing any source mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

RENDERERS_COMMIT = "c4772ac1321c69e83d2b4460600072911cc41a0b"
PRIME_RL_COMMIT = "3fc28ddfb354f336d1cc28e8e032f262f5aa68b2"

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = (
    REPOSITORY_ROOT
    / "integrations"
    / "prime_intellect"
    / "renderers-c4772-starcoder2.patch"
)
SOURCE_COPIES = {
    REPOSITORY_ROOT / "posttrain" / "prime" / "chat_format.py": Path(
        "renderers/starcoder2_chat_format.py"
    ),
    REPOSITORY_ROOT / "posttrain" / "prime" / "renderer.py": Path(
        "renderers/starcoder2_coding.py"
    ),
}
EXPECTED_SHA256 = {
    PATCH_PATH: "70b2bfdec9901094a541f55f9ec5248df657ffabbfb3e674e8d8fab862ef9565",
    REPOSITORY_ROOT
    / "posttrain"
    / "prime"
    / "chat_format.py": "36cb8ac88310928f7a90b5c119ae982fd29efd75d50ae392f0df6324d8d01446",
    REPOSITORY_ROOT
    / "posttrain"
    / "prime"
    / "renderer.py": "f66cb309ac5953f0af584823c6b9ade953848cf886ea61784310743ddcca1288",
}


class IntegrationError(RuntimeError):
    pass


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise IntegrationError(f"{' '.join(args)} failed: {detail}")
    return result


def _git(checkout: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return _run(["git", *args], cwd=checkout, check=check)


def _require_checkout(checkout: Path, *, expected_commit: str, label: str) -> None:
    if not checkout.is_dir():
        raise IntegrationError(f"{label} checkout does not exist: {checkout}")
    top = Path(_git(checkout, "rev-parse", "--show-toplevel").stdout.strip()).resolve()
    if top != checkout:
        raise IntegrationError(
            f"{label} must point to its Git root; got {checkout}, root is {top}"
        )
    actual = _git(checkout, "rev-parse", "HEAD").stdout.strip()
    if actual != expected_commit:
        raise IntegrationError(
            f"{label} HEAD must be {expected_commit}; got {actual}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _copied_sources_match(checkout: Path) -> bool:
    return all(
        destination.is_file() and _sha256(source) == _sha256(destination)
        for source, relative in SOURCE_COPIES.items()
        for destination in [checkout / relative]
    )


def _patch_is_applied(checkout: Path) -> bool:
    reverse = _git(
        checkout,
        "apply",
        "--reverse",
        "--check",
        str(PATCH_PATH),
        check=False,
    )
    return reverse.returncode == 0 and _copied_sources_match(checkout)


def _validate_python_sources() -> None:
    for path, expected in EXPECTED_SHA256.items():
        actual = _sha256(path)
        if actual != expected:
            raise IntegrationError(
                f"integration source hash mismatch for {path}: "
                f"expected {expected}, got {actual}"
            )
    for source in SOURCE_COPIES:
        text = source.read_text(encoding="utf-8")
        compile(text, str(source), "exec")


def install(
    renderers_checkout: Path,
    *,
    prime_rl_checkout: Path,
    check_only: bool,
) -> str:
    renderers_checkout = renderers_checkout.expanduser().resolve()
    prime_rl_checkout = prime_rl_checkout.expanduser().resolve()
    _require_checkout(
        renderers_checkout,
        expected_commit=RENDERERS_COMMIT,
        label="renderers",
    )
    _require_checkout(
        prime_rl_checkout,
        expected_commit=PRIME_RL_COMMIT,
        label="prime-rl",
    )
    if not PATCH_PATH.is_file():
        raise IntegrationError(f"patch is missing: {PATCH_PATH}")
    for source in SOURCE_COPIES:
        if not source.is_file():
            raise IntegrationError(f"renderer source is missing: {source}")
    _validate_python_sources()

    if _patch_is_applied(renderers_checkout):
        return "already-installed"

    status = _git(renderers_checkout, "status", "--porcelain").stdout.strip()
    if status:
        raise IntegrationError(
            "renderers checkout must be clean before installation; found:\n" + status
        )
    conflicts = [
        str(renderers_checkout / relative)
        for relative in SOURCE_COPIES.values()
        if (renderers_checkout / relative).exists()
    ]
    if conflicts:
        raise IntegrationError(
            "renderer destination files already exist but do not match: "
            + ", ".join(conflicts)
        )

    _git(renderers_checkout, "apply", "--check", str(PATCH_PATH))
    if check_only:
        return "check-passed"

    created: list[Path] = []
    patch_applied = False
    try:
        _git(renderers_checkout, "apply", str(PATCH_PATH))
        patch_applied = True
        for source, relative in SOURCE_COPIES.items():
            destination = renderers_checkout / relative
            shutil.copyfile(source, destination)
            created.append(destination)
        if not _patch_is_applied(renderers_checkout):
            raise IntegrationError("post-install verification failed")
    except Exception:
        for destination in created:
            destination.unlink(missing_ok=True)
        if patch_applied:
            _git(renderers_checkout, "apply", "--reverse", str(PATCH_PATH))
        raise
    return "installed"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderers-checkout", type=Path, required=True)
    parser.add_argument("--prime-rl-checkout", type=Path, required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="verify pins, cleanliness, sources, and patch applicability without writes",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = install(
            args.renderers_checkout,
            prime_rl_checkout=args.prime_rl_checkout,
            check_only=args.check_only,
        )
    except IntegrationError as exc:
        print(f"prime renderer integration failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"prime renderer integration: {result}; "
        f"renderers={RENDERERS_COMMIT} prime-rl={PRIME_RL_COMMIT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
