@echo off
setlocal EnableExtensions

rem Windows double-click entrypoint for the stable MuMu Decision Transformer run.
rem Keep these defaults conservative. Change them only for a labelled experiment.
set "KATACR_WSL_DISTRO=Ubuntu-24.04"
set "KATACR_BASE_RTG=3.0"
set "KATACR_MAX_ACTION_DELAY=8"

if not defined KATACR_WSL_ROOT (
  echo Set KATACR_WSL_ROOT to the WSL path of this checkout before running.
  echo Example: /mnt/d/work/KataCR-LiveLoop
  exit /b 2
)

echo.
echo === KataCR MuMu DT launcher ===
echo WSL distro: %KATACR_WSL_DISTRO%
echo Policy profile: RTG=%KATACR_BASE_RTG%, max delay=%KATACR_MAX_ACTION_DELAY%
echo.
echo Before continuing, make sure MuMu is open and ready at the training-camp entry point.
echo Press Ctrl+C now to cancel, or any key to start.
pause >nul

wsl.exe -d "%KATACR_WSL_DISTRO%" -- bash -lc "set -e; cd '%KATACR_WSL_ROOT%'; test -x ./scripts/run_mumu_dt_eval.sh; KATACR_BASE_RTG=%KATACR_BASE_RTG% KATACR_MAX_ACTION_DELAY=%KATACR_MAX_ACTION_DELAY% ./scripts/run_mumu_dt_eval.sh"
set "KATACR_EXIT_CODE=%ERRORLEVEL%"

echo.
if "%KATACR_EXIT_CODE%"=="0" (
  echo KataCR finished normally.
) else (
  echo KataCR stopped with exit code %KATACR_EXIT_CODE%.
  echo Check the messages above. Common causes are MuMu not running, ADB disconnected,
  echo or the WSL runtime mirror or Python environment being unavailable.
)
echo.
pause
exit /b %KATACR_EXIT_CODE%
