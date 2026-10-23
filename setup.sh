#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$ROOT/RVCStudio/RVCStudio.csproj"

command -v dotnet >/dev/null || {
  echo 'Install .NET SDK 10: https://dotnet.microsoft.com/download/dotnet/10.0' >&2
  exit 1
}
dotnet --list-sdks | grep -qE '^10\.' || {
  echo 'This project requires .NET SDK 10.' >&2
  exit 1
}

if ! command -v curl >/dev/null; then
  echo 'Install curl to download the base models.' >&2
  exit 1
fi
if ! command -v sha256sum >/dev/null; then
  echo 'Install sha256sum to verify downloaded models.' >&2
  exit 1
fi
if ! ldconfig -p 2>/dev/null | grep -qi 'libportaudio'; then
  echo 'Warning: system PortAudio library not found (for example, libportaudio2).' >&2
fi

echo '[1/4] Downloading and verifying base ONNX models...'
DEST="$ROOT/RVCStudio/assets/models"
REPO="https://huggingface.co/wok000/weights_gpl/resolve/d1b578cfcdf944dd3910786f7604754c79234b36"
mkdir -p "$DEST"
download_model() {
  local path="$1" name="$2" hash="$3" target="$DEST/$2"
  if [[ -f "$target" ]] && echo "$hash  $target" | sha256sum --check --status; then return; fi
  curl --fail --location --retry 3 --continue-at - "$REPO/$path?download=true" --output "$target.part"
  echo "$hash  $target.part" | sha256sum --check --status || { rm -f "$target.part"; echo "SHA-256 verification failed: $name" >&2; exit 1; }
  mv "$target.part" "$target"
}
download_model 'content-vec/contentvec-f.onnx' contentvec.onnx 4b31ed3d95a568fab7952de923ff7f7d3d17128ea6fce69f665509d24c3156db
download_model 'rmvpe/rmvpe_20231006.onnx' rmvpe.onnx 84f0586308e36157f75b77c8591bf636d6719c0c4ba95f8faf3df479e7566219
for notice in LICENSE THIRD_PARTY_NOTICES.md; do curl --fail --location --retry 3 "$REPO/$notice?download=true" --output "$DEST/$notice"; done

echo '[2/4] Building the standalone Python converter...'
CONVERTER="$ROOT/RVCStudio/converter"
VENV="$CONVERTER/.venv-build"
[[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$CONVERTER/requirements.txt" pyinstaller
(cd "$CONVERTER" && "$VENV/bin/python" -m PyInstaller --clean --noconfirm --distpath "$CONVERTER/dist/linux-x64" build.spec)
test -f "$CONVERTER/dist/linux-x64/rvc-converter"
chmod +x "$CONVERTER/dist/linux-x64/rvc-converter"

echo '[3/4] Restoring .NET dependencies...'
dotnet restore "$ROOT/RVCStudio.sln"

echo '[4/4] Building the application...'
dotnet build "$PROJECT" -c Debug --no-restore

echo 'Setup complete. Run: dotnet run --project RVCStudio/RVCStudio.csproj'
