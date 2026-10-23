$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) { throw 'Install .NET SDK 10: https://dotnet.microsoft.com/download/dotnet/10.0' }
if (-not ((dotnet --list-sdks) -match '^10\.')) { throw 'This project requires .NET SDK 10.' }
if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) { throw 'curl.exe is required to download models.' }

$Destination = Join-Path $Root 'RVCStudio/assets/models'
$Repository = 'https://huggingface.co/wok000/weights_gpl/resolve/d1b578cfcdf944dd3910786f7604754c79234b36'
New-Item -ItemType Directory -Force -Path $Destination | Out-Null
function Get-Model([string]$Path, [string]$Name, [string]$Hash) {
    $Target = Join-Path $Destination $Name
    if (Test-Path $Target) {
        if ((Get-FileHash -Algorithm SHA256 $Target).Hash.ToLowerInvariant() -eq $Hash) { return }
    }
    $Temporary = "$Target.part"
    & curl.exe --fail --location --retry 3 --continue-at - "$Repository/$Path`?download=true" --output $Temporary
    if ($LASTEXITCODE -ne 0) { throw "Could not download $Name" }
    if ((Get-FileHash -Algorithm SHA256 $Temporary).Hash.ToLowerInvariant() -ne $Hash) {
        Remove-Item -Force $Temporary
        throw "SHA-256 verification failed: $Name"
    }
    Move-Item -Force $Temporary $Target
}

Write-Host '[1/4] Downloading and verifying base ONNX models...'
Get-Model 'content-vec/contentvec-f.onnx' 'contentvec.onnx' '4b31ed3d95a568fab7952de923ff7f7d3d17128ea6fce69f665509d24c3156db'
Get-Model 'rmvpe/rmvpe_20231006.onnx' 'rmvpe.onnx' '84f0586308e36157f75b77c8591bf636d6719c0c4ba95f8faf3df479e7566219'
foreach ($Notice in @('LICENSE', 'THIRD_PARTY_NOTICES.md')) {
    & curl.exe --fail --location --retry 3 "$Repository/$Notice`?download=true" --output (Join-Path $Destination $Notice)
    if ($LASTEXITCODE -ne 0) { throw "Could not download $Notice" }
}

Write-Host '[2/4] Building the standalone Python converter...'
$Converter = Join-Path $Root 'RVCStudio/converter'
$Venv = Join-Path $Converter '.venv-build'
$Python = Join-Path $Venv 'Scripts/python.exe'
if (-not (Test-Path $Python)) { & python -m venv $Venv; if ($LASTEXITCODE -ne 0) { throw 'Could not create the converter environment.' } }
& $Python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not update pip.' }
& $Python -m pip install -r (Join-Path $Converter 'requirements.txt') pyinstaller
if ($LASTEXITCODE -ne 0) { throw 'Could not install converter dependencies.' }
Push-Location $Converter
try { & $Python -m PyInstaller --clean --noconfirm --distpath (Join-Path $Converter 'dist/windows-x64') build.spec }
finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { throw 'Could not build the Python converter.' }
if (-not (Test-Path (Join-Path $Converter 'dist/windows-x64/rvc-converter.exe'))) { throw 'PyInstaller did not create the converter.' }
Write-Host '[3/4] Restoring .NET dependencies...'
dotnet restore (Join-Path $Root 'RVCStudio.sln')
if ($LASTEXITCODE -ne 0) { throw 'Could not restore .NET dependencies.' }
Write-Host '[4/4] Building the application...'
dotnet build (Join-Path $Root 'RVCStudio/RVCStudio.csproj') -c Debug --no-restore
if ($LASTEXITCODE -ne 0) { throw 'Could not build the application.' }
Write-Host 'Setup complete. Run: dotnet run --project RVCStudio/RVCStudio.csproj'
