# Run from PowerShell as administrator.
#Requires -RunAsAdministrator

$ErrorActionPreference = 'Stop'
$distro = 'Ubuntu-22.04'

$installed = @(
    & wsl.exe --list --quiet 2>$null |
        ForEach-Object { ($_ -replace "`0", '').Trim() }
)

if ($installed -contains $distro) {
    Write-Host "$distro is already installed."
} else {
    & wsl.exe --install -d $distro

    if ($LASTEXITCODE -notin @(0, 3010)) {
        throw 'WSL installation did not finish. Follow its displayed instructions, then rerun this script.'
    }
}

Write-Host ''
Write-Host 'If Windows requests a restart, save your work and restart.'
Write-Host "Then open Ubuntu with: wsl -d $distro"
Write-Host 'Complete Linux username/password setup if prompted.'
Write-Host 'Navigate to this project in Ubuntu, then run:'
Write-Host 'bash setup-brain.sh'