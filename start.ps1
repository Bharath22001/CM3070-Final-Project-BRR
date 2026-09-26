# start.ps1 -- one command to open a working session for this project.
#
# Run from D:\FYP with:    .\start.ps1          (CPU env: Domains 2 and 3)
#                          .\start.ps1 -Rocm    (GPU env: Domain 1 training)
#
# WHY THERE ARE TWO ENVIRONMENTS
# fyp_env holds torch 2.14.0+cpu, which Domains 2 and 3 run against.
# fyp_env_rocm holds torch 2.9.1+rocm7.2.1 for training Domain 1 on the
# RX 7900 XT. They are kept separate because installing the ROCm torch into
# fyp_env would downgrade the torch underneath Domains 2 and 3.
#
# WHY THIS EXISTS
# Setting HF_HOME as a persistent Windows user variable does not reach the
# VS Code terminal, because processes inherit their environment from
# explorer.exe, which caches its copy at login. Setting the variables here
# makes the project self-contained: it does not depend on machine state a
# future reader (or a different machine) would not have.
#
# WHY IT MATTERS: without HF_HOME, HuggingFace writes model downloads to
# C:\Users\<name>\.cache -- the drive that just ran out of space.

param([switch]$Rocm)

Set-Location D:\FYP

Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned -Force

if ($Rocm) {
    & D:\FYP\fyp_env_rocm\Scripts\Activate.ps1
} else {
    & D:\FYP\fyp_env\Scripts\Activate.ps1
}

$env:HF_HOME       = 'D:\FYP\.hf_cache'
$env:PIP_CACHE_DIR = 'D:\FYP\.pip_cache'

# C: runs close to full. pip unpacks wheels through %TEMP%, so keep that on D:
# too -- otherwise a large install fails on free space rather than on anything
# to do with the package.
$env:TMP  = 'D:\FYP\.tmp'
$env:TEMP = 'D:\FYP\.tmp'

New-Item -ItemType Directory -Force -Path $env:HF_HOME, $env:PIP_CACHE_DIR, $env:TMP | Out-Null

Write-Host ""
Write-Host "FYP session ready" -ForegroundColor Green
Write-Host "  env          : $(if ($Rocm) { 'fyp_env_rocm (GPU / Domain 1)' } else { 'fyp_env (CPU / Domains 2-3)' })"
Write-Host "  python       : $((Get-Command python).Source)"
Write-Host "  HF_HOME      : $env:HF_HOME"
Write-Host "  PIP_CACHE_DIR: $env:PIP_CACHE_DIR"
Write-Host "  TMP/TEMP     : $env:TMP"
Write-Host ""