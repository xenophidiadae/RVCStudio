$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -m pip install -r requirements.txt pyinstaller
pyinstaller --clean --noconfirm --onefile --name rvc-converter --distpath dist/windows-x64 build.spec
