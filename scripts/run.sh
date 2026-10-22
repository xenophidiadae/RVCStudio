#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec dotnet run --project "$ROOT/RVCStudio/RVCStudio.csproj" "$@"
