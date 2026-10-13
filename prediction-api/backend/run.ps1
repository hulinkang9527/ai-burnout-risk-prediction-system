$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $venvPython) {
    & $venvPython -m uvicorn main:app --host 127.0.0.1 --port 8000
    exit $LASTEXITCODE
}

if (Get-Command python -ErrorAction SilentlyContinue) {
    python -m uvicorn main:app --host 127.0.0.1 --port 8000
    exit $LASTEXITCODE
}

if (Get-Command py -ErrorAction SilentlyContinue) {
    py -3 -m uvicorn main:app --host 127.0.0.1 --port 8000
    exit $LASTEXITCODE
}

throw "未找到 Python。请先安装 Python 3.11+，并按 README 创建 .venv。"
