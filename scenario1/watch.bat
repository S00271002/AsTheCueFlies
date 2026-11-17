@echo off
cd /d "%~dp0"
wsl.exe -d Ubuntu-22.04 --cd "%~dp0." -- bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate brain-fly && python watch.py"
pause
