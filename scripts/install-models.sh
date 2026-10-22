#!/usr/bin/env bash
# Downloads and verifies the shared ContentVec and RMVPE ONNX models.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$ROOT/download-models.sh"
