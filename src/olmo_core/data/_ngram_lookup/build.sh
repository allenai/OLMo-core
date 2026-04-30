#!/usr/bin/env bash
# Build libngram_lookup.so for use in olmo-core training jobs.
#
# Steps (idempotent — re-running is safe):
#   1. Clone KenLM at the pinned commit if not already present.
#   2. Apply our small `lm/model.hh` patch (idempotent).
#   3. cmake + make on kenlm to produce libkenlm.a + libkenlm_util.a.
#   4. Compile lookup.cc, link against the kenlm static libs, output .so.
#
# Designed for the olmo-core training container (Ubuntu / apt-based).
# Boost / cmake / build tools are installed in the standard image; if any
# are missing the script aborts with an explicit message.
#
# Inputs (env or default):
#   KENLM_SRC    path to (or destination of) kenlm checkout.
#                Default: <out_dir>/kenlm
#   OUT_DIR      where to put kenlm + libngram_lookup.so.
#                Default: /tmp/olmo_core_ngram_lookup
#   KENLM_REF    pinned kenlm commit. Default: 4cb443e60b7bf2c0ddf3c745378f76cb59e254e5
#
# Output:
#   $OUT_DIR/libngram_lookup.so

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PATCH_FILE="$SCRIPT_DIR/kenlm_expose_search.patch"
LOOKUP_CC="$SCRIPT_DIR/lookup.cc"

OUT_DIR="${OUT_DIR:-/tmp/olmo_core_ngram_lookup}"
KENLM_SRC="${KENLM_SRC:-$OUT_DIR/kenlm}"
KENLM_REF="${KENLM_REF:-4cb443e60b7bf2c0ddf3c745378f76cb59e254e5}"

mkdir -p "$OUT_DIR"

echo "=== build.sh config ==="
printf 'OUT_DIR=%s\nKENLM_SRC=%s\nKENLM_REF=%s\nPATCH=%s\nLOOKUP_CC=%s\n' \
    "$OUT_DIR" "$KENLM_SRC" "$KENLM_REF" "$PATCH_FILE" "$LOOKUP_CC"
echo

[ -f "$PATCH_FILE" ] || { echo "PATCH_FILE not found: $PATCH_FILE" >&2; exit 1; }
[ -f "$LOOKUP_CC" ] || { echo "LOOKUP_CC not found: $LOOKUP_CC" >&2; exit 1; }

# Best-effort install of build deps. The olmo-core training image is
# Ubuntu-based; cmake/g++/boost are typically already there. apt-get
# install is a no-op if they're present.
SUDO=""; [ "$EUID" -ne 0 ] && SUDO=sudo
if command -v apt-get >/dev/null 2>&1; then
    if ! command -v cmake >/dev/null || ! ls /usr/include/boost/version.hpp >/dev/null 2>&1; then
        echo "=== installing build deps via apt-get ==="
        DEBIAN_FRONTEND=noninteractive $SUDO apt-get update -qq
        DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -y -qq --no-install-recommends \
            g++ cmake libboost-all-dev build-essential git zlib1g-dev libbz2-dev liblzma-dev \
            > /tmp/apt.log 2>&1 || { echo "apt-get install failed; tail of log:" >&2; tail /tmp/apt.log >&2; exit 1; }
    fi
fi
command -v cmake >/dev/null || { echo "cmake still not found" >&2; exit 1; }
command -v c++ >/dev/null || { echo "c++ still not found" >&2; exit 1; }

# 1) Clone kenlm at pinned commit (idempotent).
if [ ! -d "$KENLM_SRC/.git" ]; then
    echo "=== cloning kenlm @ $KENLM_REF ==="
    git clone --quiet https://github.com/kpu/kenlm.git "$KENLM_SRC"
    (cd "$KENLM_SRC" && git checkout --quiet "$KENLM_REF")
elif [ "$(cd "$KENLM_SRC" && git rev-parse HEAD)" != "$KENLM_REF" ]; then
    echo "=== resetting kenlm @ $KENLM_REF ==="
    (cd "$KENLM_SRC" && git fetch --quiet --depth 1 origin "$KENLM_REF" 2>/dev/null || git fetch --quiet origin "$KENLM_REF")
    (cd "$KENLM_SRC" && git checkout --quiet "$KENLM_REF")
fi
echo

# 2) Apply patch idempotently.
echo "=== applying patch (idempotent) ==="
if (cd "$KENLM_SRC" && git apply --check "$PATCH_FILE" 2>/dev/null); then
    (cd "$KENLM_SRC" && git apply "$PATCH_FILE")
    echo "patch applied."
elif (cd "$KENLM_SRC" && git apply --reverse --check "$PATCH_FILE" 2>/dev/null); then
    echo "patch already applied — skipping."
else
    echo "patch does not apply cleanly and is not already applied" >&2
    (cd "$KENLM_SRC" && git status --short) >&2
    exit 1
fi
echo

# 3) Build kenlm (only kenlm + kenlm_util; we don't need lmplz / build_binary
#    inside the training container).
KENLM_BUILD="$KENLM_SRC/build"
mkdir -p "$KENLM_BUILD"

# If libkenlm.a already exists and is newer than the patch file, skip cmake.
if [ -f "$KENLM_BUILD/lib/libkenlm.a" ] && [ "$KENLM_BUILD/lib/libkenlm.a" -nt "$PATCH_FILE" ]; then
    echo "=== libkenlm.a up to date — skipping cmake/make ==="
else
    echo "=== cmake + make kenlm ==="
    T0=$(date +%s)
    # On Apple Silicon, brew installs boost@1.85 keg-only (current `boost`
    # 1.90 dropped the separately-compiled libboost_system that kenlm
    # requires). Hint at the older formula. Harmless on Linux.
    BOOST_HINT="${BOOST_HINT:-/opt/homebrew/opt/boost@1.85}"
    CMAKE_PREFIX_FLAG=""
    [ -d "$BOOST_HINT" ] && CMAKE_PREFIX_FLAG="-DCMAKE_PREFIX_PATH=$BOOST_HINT"
    (cd "$KENLM_BUILD" && cmake .. -DCMAKE_BUILD_TYPE=Release $CMAKE_PREFIX_FLAG > /tmp/kenlm_cmake.log 2>&1) \
        || { echo "kenlm cmake failed; tail of log:" >&2; tail -30 /tmp/kenlm_cmake.log >&2; exit 1; }
    NPROC=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
    (cd "$KENLM_BUILD" && make -j"$NPROC" kenlm kenlm_util > /tmp/kenlm_make.log 2>&1) \
        || { echo "kenlm make failed; tail of log:" >&2; tail -30 /tmp/kenlm_make.log >&2; exit 1; }
    echo "kenlm build: $(( $(date +%s) - T0 ))s"
fi
echo

# 4) Compile + link lookup.cc.
echo "=== compile libngram_lookup.so ==="
T0=$(date +%s)

UNAME=$(uname -s)
CXX="${CXX:-c++}"
COMMON_FLAGS="-O2 -fPIC -std=c++17 -I$KENLM_SRC -DKENLM_MAX_ORDER=6"
LINK_LIBS="-lz -lbz2 -llzma -lpthread"
KENLM_LIBS="$KENLM_BUILD/lib/libkenlm.a $KENLM_BUILD/lib/libkenlm_util.a"

if [ "$UNAME" = "Darwin" ]; then
    SO_FLAGS="-dynamiclib -undefined dynamic_lookup"
    SO_NAME="libngram_lookup.dylib"
else
    SO_FLAGS="-shared"
    SO_NAME="libngram_lookup.so"
fi

"$CXX" $COMMON_FLAGS $SO_FLAGS \
    "$LOOKUP_CC" \
    $KENLM_LIBS \
    $LINK_LIBS \
    -o "$OUT_DIR/$SO_NAME"

echo "compile: $(( $(date +%s) - T0 ))s"
ls -lh "$OUT_DIR/$SO_NAME"

# Symlink so .so is always present for the Python wrapper's resolver.
[ "$SO_NAME" != "libngram_lookup.so" ] && ln -sf "$SO_NAME" "$OUT_DIR/libngram_lookup.so"
echo

echo "=== DONE ==="
echo "extension at: $OUT_DIR/$SO_NAME"
echo "set OLMO_NGRAM_LOOKUP_DYLIB=$OUT_DIR/$SO_NAME if not at the default location"
