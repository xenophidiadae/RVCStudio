$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) { throw 'Установите .NET SDK 10: https://dotnet.microsoft.com/download/dotnet/10.0' }
if (-not ((dotnet --list-sdks) -match '^10\.')) { throw 'Для проекта требуется .NET SDK 10.' }
& (Join-Path $Root 'download-models.ps1')
if ($LASTEXITCODE -ne 0) { throw 'Не удалось загрузить базовые модели.' }
dotnet restore (Join-Path $Root 'RVCStudio.sln')
if ($LASTEXITCODE -ne 0) { throw 'Не удалось восстановить зависимости.' }
Write-Host 'Настройка завершена. Запустите .\scripts\run.ps1'
