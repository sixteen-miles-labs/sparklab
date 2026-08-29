#!/usr/bin/env python3
"""Retag native Linux runtime wheels to their detected manylinux policy.

setuptools tags C-extension wheels `linux_<arch>`, which PyPI rejects at upload.
The CI build already runs inside the pytorch manylinux_2_28 container, so the
binaries meet the policy — only the tag is missing. This asks auditwheel which
policy the wheel's symbol versions actually satisfy and rewrites the tag.

Two deliberate choices:
  * `sym_policy`, not the overall policy: the extensions intentionally leave
    libtorch/libcudart as external NEEDED entries (provided by the installed
    torch), and the overall policy grades any non-whitelisted external lib as
    plain `linux`. `auditwheel repair` is equally unusable here — it would try
    to graft libtorch into the wheel. Detection approach follows vLLM's
    detect-manylinux-tag.py (Apache-2.0).
  * Detected tag, not a hard-coded one, with a ceiling: a wheel built outside
    the container (host glibc 2.3x) must fail loudly instead of shipping a tag
    that lies about its glibc floor.

`wheel tags` rewrites filename + WHEEL metadata + RECORD consistently; renaming
the file alone would leave the wheel internally inconsistent.

Requires: auditwheel==6.6.0 (the analyze_wheel_abi signature is version
specific), wheel.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from auditwheel.error import NonPlatformWheelError, WheelToolsError
from auditwheel.wheel_abi import analyze_wheel_abi
from auditwheel.wheeltools import get_wheel_architecture, get_wheel_libc
from packaging.utils import InvalidWheelFilename


def detect_platform_tag(wheel: Path) -> str:
    try:
        arch = get_wheel_architecture(wheel.name)
    except (WheelToolsError, NonPlatformWheelError, InvalidWheelFilename):
        arch = None
    try:
        libc = get_wheel_libc(wheel.name)
    except (WheelToolsError, InvalidWheelFilename):
        libc = None
    winfo = analyze_wheel_abi(
        libc,
        arch,
        wheel,
        frozenset(),
        disable_isa_ext_check=False,
        allow_graft=False,
    )
    # Deliberately not winfo.overall_policy: that folds in the external-library
    # check, and libtorch/libcudart are not on any manylinux whitelist, so it
    # always collapses to a plain linux platform tag. sym_policy (glibc symbols) and
    # machine_policy (required ISA extensions) are the two components that do
    # apply to us. min() by priority picks the stricter one -- the same
    # combinator auditwheel itself uses -- so a future global -march= that
    # raises the ISA floor cannot be silently retagged as broadly compatible.
    return min(winfo.sym_policy, winfo.machine_policy).name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument(
        "--max-glibc",
        default="2.28",
        help="highest glibc version the detected tag may require (default: 2.28)",
    )
    args = parser.parse_args()
    m = re.fullmatch(r"(\d+)\.(\d+)", args.max_glibc)
    if not m:
        print(f"error: --max-glibc must look like '2.28' (got '{args.max_glibc}')", file=sys.stderr)
        return 1
    ceiling = (int(m[1]), int(m[2]))

    for whl in args.wheels:
        if not whl.is_file():
            print(f"error: no such wheel: {whl}", file=sys.stderr)
            return 1
        tag = detect_platform_tag(whl)
        m = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", tag)
        if not m or (int(m[1]), int(m[2])) > ceiling:
            print(
                f"error: {whl.name}: detected policy '{tag}' exceeds "
                f"manylinux_{ceiling[0]}_{ceiling[1]} — this wheel was not "
                "built in the manylinux container and must not be retagged",
                file=sys.stderr,
            )
            return 2
        subprocess.run(
            [sys.executable, "-m", "wheel", "tags", "--platform-tag", tag, "--remove", str(whl)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        print(f"{whl.name} -> {tag}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
