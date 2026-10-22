#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
command -v dotnet >/dev/null || { echo 'Установите .NET SDK 10: https://dotnet.microsoft.com/download/dotnet/10.0' >&2; exit 1; }
dotnet --list-sdks | grep -qE '^10\.' || { echo 'Для проекта требуется .NET SDK 10.' >&2; exit 1; }
if ! ldconfig -p 2>/dev/null | grep -qi 'libportaudio'; then
  echo 'Предупреждение: не найдена системная библиотека PortAudio (например, libportaudio2).' >&2
fi
"$ROOT/download-models.sh"
dotnet restore "$ROOT/RVCStudio.sln"
echo 'Настройка завершена. Запустите ./scripts/run.sh'
