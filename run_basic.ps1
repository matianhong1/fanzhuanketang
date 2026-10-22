param([string]$Python = 'F:\Anaconda3\python.exe', [switch]$Full)
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = 'utf-8'
Set-Location -LiteralPath $PSScriptRoot
& $Python scripts/download_data.py
if ($LASTEXITCODE -ne 0) { throw '数据准备失败，查看下载说明' }
if ($Full) { & $Python main.py all } else { & $Python main.py all --quick }
if ($LASTEXITCODE -ne 0) { throw '实验失败，检查运行输出' }
if ($Full) { $resultPath = 'outputs/full' } else { $resultPath = 'outputs/quick' }
& $Python scripts/analyze_results.py --output $resultPath
if ($LASTEXITCODE -ne 0) { throw '结果分析失败' }
& $Python scripts/verify_results.py --output $resultPath
if ($LASTEXITCODE -ne 0) { throw '交付检查失败' }
