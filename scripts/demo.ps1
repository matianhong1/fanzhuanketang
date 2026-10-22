param([string]$Python = 'F:\Anaconda3\python.exe', [string]$Run = 'full')
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
foreach ($kind in @('success', 'failure')) {
    $demoInput = Get-ChildItem -LiteralPath "outputs/$Run/demo_inputs" -Filter "$($kind)_*.png" | Sort-Object Name | Select-Object -First 1
    if ($null -eq $demoInput) { throw "No $kind example found; first run main.py all" }
    & $Python main.py predict --image $demoInput.FullName --model "outputs/$Run/models/hog_svm.joblib" --output "outputs/$Run/demo_$kind.png"
    if ($LASTEXITCODE -ne 0) { throw 'Prediction failed' }
}
