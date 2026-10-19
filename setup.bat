@echo off
setlocal
cd /d "%~dp0"

echo Setting up WSL and Ubuntu...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-windows.ps1"

if errorlevel 1 (
    echo.
    echo Windows setup did not complete.
    echo Check the message above.
    pause
    exit /b 1
)

echo.
echo Running Linux setup...
wsl.exe -d Ubuntu-22.04 -- bash -lc "cd -- \"$(wslpath -u \"$1\")\" && bash setup-brain.sh" bash "%~dp0."

if errorlevel 1 (
    echo.
    echo Linux setup did not complete.
    echo Check the message above. Restart only if requested.
    pause
    exit /b 1
)

echo.
echo Opening the project with brain-fly activated...
wsl.exe -d Ubuntu-22.04 --cd "%~dp0." -- bash -i -c "source ~/miniconda3/etc/profile.d/conda.sh && conda activate brain-fly && exec bash -i"

if errorlevel 1 (
    echo Could not open the project environment.
    pause
    exit /b 1
)