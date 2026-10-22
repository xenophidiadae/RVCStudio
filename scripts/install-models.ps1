$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
& (Join-Path $Root 'download-models.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Не удалось установить базовые ONNX-модели.' }
