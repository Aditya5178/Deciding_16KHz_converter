@echo off
setlocal
cd /d "%~dp0"

REM Drag one WAV/MP3 file or one folder of audio files onto this file.
set "INPUT=%~1"
if "%INPUT%"=="" (
    echo.
    echo Drag a WAV file or a folder containing WAV files onto this batch file.
    set /p "INPUT=Or paste the full path here: "
)
set "INPUT=%INPUT:"=%"

if not exist "%INPUT%" (
    echo.
    echo ERROR: The input file or folder was not found:
    echo %INPUT%
    pause
    exit /b 1
)

where py >nul 2>nul
if errorlevel 1 (
    echo.
    echo Python is required but was not found.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo During installation, select "Add Python to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Setting up Python environment. This happens only once...
    py -3 -m venv .venv
    if errorlevel 1 goto :failed
    .venv\Scripts\python.exe -m pip install --upgrade pip
    if errorlevel 1 goto :failed
    .venv\Scripts\python.exe -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)

if exist "%INPUT%\NUL" (
    set "OUTPUT=%INPUT%_16k"
    set "OUTPUT_TYPE=folder"
) else (
    for %%I in ("%INPUT%") do set "OUTPUT=%%~dpnI_16k.wav"
    set "OUTPUT_TYPE=file"
)

echo.
echo Converting to 16 kHz mono WAV...
echo Output: %OUTPUT%
.venv\Scripts\python.exe resample_to_16k.py "%INPUT%" "%OUTPUT%"
if errorlevel 1 goto :failed

echo.
echo Done. Converted %OUTPUT_TYPE%: %OUTPUT%
pause
exit /b 0

:failed
echo.
echo Conversion failed. See the message above for details.
pause
exit /b 1
