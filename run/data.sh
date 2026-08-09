#!/usr/bin/env bash
# Build task data.  All arguments are passed through to `ctc-data`.
#
#   run/data.sh list
#   run/data.sh build --task contradiction --rungs 2k,4k,8k --out /data/ctc/v3
#   run/data.sh audit --task contradiction --dir /data/ctc/v3
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
exec "$CTC_PYTHON" -m ctc.data.cli "$@"
