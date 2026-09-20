"""Fail closed unless both FLA distributions and the int64 history-offset kernel match."""

import hashlib
from importlib.metadata import distribution, version

EXPECTED_SHA256 = "e78c79c5889148fd471e7ec770800da4e58bce2cc40c9c1fcebf86a4dd72a2e2"


def main():
    for package in ("flash-linear-attention", "fla-core"):
        installed = version(package)
        print(f"[FLA verification] {package}={installed}", flush=True)
        if installed != "0.4.2":
            raise RuntimeError(f"Expected {package}==0.4.2, found {installed}")
    path = distribution("fla-core").locate_file("fla/ops/common/chunk_delta_h.py")
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    print(f"[FLA verification] {path} sha256={digest}", flush=True)
    if digest != EXPECTED_SHA256 or b"i_t.to(tl.int64)" not in source:
        raise RuntimeError("Unverified FLA history-pointer kernel; refusing evaluation")


if __name__ == "__main__":
    main()
