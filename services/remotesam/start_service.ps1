param(
    [string]$EnvironmentName = "geoscope-remotesam",
    [string]$CheckpointPath = $env:GEO_REMOTESAM_CHECKPOINT,
    [string]$TokenizerPath = $env:GEO_REMOTESAM_TOKENIZER,
    [string]$BindAddress = "0.0.0.0",
    [int]$Port = 6657,
    [string]$CacheRoot = $env:GEO_REMOTESAM_CACHE_ROOT
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "upstream")).Path

if (-not $CheckpointPath -or -not (Test-Path -LiteralPath $CheckpointPath -PathType Leaf)) {
    throw "Pass -CheckpointPath with the downloaded RemoteSAMv1.pth file."
}
if (-not $TokenizerPath -or -not (Test-Path -LiteralPath $TokenizerPath -PathType Container)) {
    throw "Pass -TokenizerPath with a local bert-base-uncased tokenizer directory."
}
if (-not $CacheRoot) {
    $CacheRoot = Join-Path $env:LOCALAPPDATA "GeoScope\RemoteSAM"
}

$env:GEO_REMOTESAM_CHECKPOINT = (Resolve-Path -LiteralPath $CheckpointPath).Path
$env:GEO_REMOTESAM_TOKENIZER = (Resolve-Path -LiteralPath $TokenizerPath).Path
$env:GEO_REMOTESAM_DEVICE = "cuda:0"
$env:GEO_REMOTESAM_HOST = $BindAddress
$env:GEO_REMOTESAM_PORT = [string]$Port
$env:GEO_REMOTESAM_FP16 = "1"
$env:GEO_REMOTESAM_USE_EPOC = "0"
$env:CUPY_CACHE_DIR = Join-Path $CacheRoot "cupy-cache"
New-Item -ItemType Directory -Path $env:CUPY_CACHE_DIR -Force | Out-Null

Set-Location -LiteralPath $projectRoot
conda run --no-capture-output -n $EnvironmentName python -u remotesam.py
