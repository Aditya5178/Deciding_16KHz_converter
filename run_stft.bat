@echo off
setlocal
cd /d "%~dp0"

REM Drag any WAV file onto this script, or paste its path when prompted.
set "INPUT=%~1"
if "%INPUT%"=="" (
    echo.
    echo Drag a WAV file onto this batch file.
    set /p "INPUT=Or paste the full path here: "
)
set "INPUT=%INPUT:"=%"

if not exist "%INPUT%" (
    echo.
    echo ERROR: WAV file not found:
    echo %INPUT%
    echo.
    echo Drag a WAV file onto run_stft.bat, or paste a valid WAV path.
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

echo.
echo Creating spectrogram...
.venv\Scripts\python.exe stft_fft.py "%INPUT%" --show
if errorlevel 1 goto :failed

echo.
echo Done. The PNG file is saved beside the WAV file.
pause
exit /b 0

:failed
echo.
echo Setup or spectrogram creation failed. See the message above for details.
pause
exit /b 1
