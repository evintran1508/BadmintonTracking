@echo off
setlocal enabledelayedexpansion
if "%1"=="test" (
    python -m unittest discover -s tests -p "test_*.py" -v
    exit /b !ERRORLEVEL!
)
if "%1"=="lint" (
    python -m compileall -q schema fixtures ai_core backend eval scripts tests
    exit /b !ERRORLEVEL!
)
if "%1"=="setup" (
    python -m scripts.setup_env
    exit /b !ERRORLEVEL!
)
if "%1"=="weights" (
    python -m scripts.download_weights
    exit /b !ERRORLEVEL!
)
echo Unknown target: %1
exit /b 1
