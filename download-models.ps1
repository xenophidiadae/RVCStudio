$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
$Destination = Join-Path $Root "RVCStudio/assets/models"
$Repository = "https://huggingface.co/wok000/weights_gpl/resolve/d1b578cfcdf944dd3910786f7604754c79234b36"
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

function Download-AndVerify([string]$RelativePath, [string]$FileName, [string]$ExpectedHash) {
    $Target = Join-Path $Destination $FileName
    $Temporary = "$Target.part"

    if (Test-Path $Target) {
        $ExistingHash = (Get-FileHash -Algorithm SHA256 $Target).Hash.ToLowerInvariant()
        if ($ExistingHash -eq $ExpectedHash) {
            Write-Host "Already present and verified: $FileName"
            return
        }
    }

    Write-Host "Downloading $FileName (large file; this may take a while)..."
    & curl.exe --fail --location --retry 3 --continue-at - `
        "$Repository/$RelativePath`?download=true" --output $Temporary
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $FileName" }

    $DownloadedHash = (Get-FileHash -Algorithm SHA256 $Temporary).Hash.ToLowerInvariant()
    if ($DownloadedHash -ne $ExpectedHash) {
        Remove-Item -Force $Temporary
        throw "SHA-256 verification failed for $FileName"
    }
    Move-Item -Force $Temporary $Target
    Write-Host "Verified: $Target"
}

Download-AndVerify `
    "content-vec/contentvec-f.onnx" `
    "contentvec.onnx" `
    "4b31ed3d95a568fab7952de923ff7f7d3d17128ea6fce69f665509d24c3156db"

Download-AndVerify `
    "rmvpe/rmvpe_20231006.onnx" `
    "rmvpe.onnx" `
    "84f0586308e36157f75b77c8591bf636d6719c0c4ba95f8faf3df479e7566219"

foreach ($Notice in @("LICENSE", "THIRD_PARTY_NOTICES.md")) {
    & curl.exe --fail --location --retry 3 "$Repository/$Notice`?download=true" --output (Join-Path $Destination $Notice)
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $Notice" }
}

Write-Host "Downloaded w-okada-compatible RVC assets. License: GPL-3.0; see RVCStudio/assets/models/THIRD_PARTY_NOTICES.md."
