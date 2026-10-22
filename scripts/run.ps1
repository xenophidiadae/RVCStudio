$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
dotnet run --project (Join-Path $Root 'RVCStudio/RVCStudio.csproj') @args
exit $LASTEXITCODE
