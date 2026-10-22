#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/RVCStudio/assets/models"
REPO="https://huggingface.co/wok000/weights_gpl/resolve/d1b578cfcdf944dd3910786f7604754c79234b36"
mkdir -p "$DEST"

download_and_verify() {
    local relative_path="$1"
    local filename="$2"
    local expected_sha256="$3"
    local target="$DEST/$filename"
    local temporary="$target.part"

    if [[ -f "$target" ]] && echo "$expected_sha256  $target" | sha256sum --check --status; then
        echo "Already present and verified: $filename"
        return
    fi

    echo "Downloading $filename (large file; this may take a while)..."
    curl --fail --location --retry 3 --continue-at - \
        "$REPO/$relative_path?download=true" --output "$temporary"
    echo "$expected_sha256  $temporary" | sha256sum --check --status || {
        rm -f "$temporary"
        echo "SHA-256 verification failed for $filename" >&2
        exit 1
    }
    mv "$temporary" "$target"
    echo "Verified: $target"
}

download_and_verify \
    "content-vec/contentvec-f.onnx" \
    "contentvec.onnx" \
    "4b31ed3d95a568fab7952de923ff7f7d3d17128ea6fce69f665509d24c3156db"

download_and_verify \
    "rmvpe/rmvpe_20231006.onnx" \
    "rmvpe.onnx" \
    "84f0586308e36157f75b77c8591bf636d6719c0c4ba95f8faf3df479e7566219"

for notice in LICENSE THIRD_PARTY_NOTICES.md; do
    curl --fail --location --retry 3 "$REPO/$notice?download=true" --output "$DEST/$notice"
done

echo "Downloaded w-okada-compatible RVC assets. License: GPL-3.0; see RVCStudio/assets/models/THIRD_PARTY_NOTICES.md."
