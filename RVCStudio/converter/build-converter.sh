#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python -m pip install -r requirements.txt pyinstaller
pyinstaller --clean --noconfirm --onefile --name rvc-converter --distpath dist/linux-x64 build.spec
