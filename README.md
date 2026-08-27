# Portable STFT spectrogram generator

## One-time requirement

Install Python 3.10 or newer from [python.org](https://www.python.org/downloads/). During installation, select **Add Python to PATH**.

## Run

1. Double-click `run_stft.bat`.
2. Paste the full path of your WAV file, or drag the WAV file onto `run_stft.bat`.

You can also drag any WAV file onto `run_stft.bat`. On its first use, the launcher creates a private `.venv` environment and downloads the Python packages. It opens the plot and also saves `<input-name>_spectrogram.png` next to the input WAV file.

## Optional command-line use

```bat
.venv\Scripts\python.exe stft_fft.py "C:\path\to\your\audio.wav" --max-frequency 8000 --show
```

`--window-size` (default `400`) controls time versus frequency detail. A smaller value gives better time detail; a larger value gives better frequency detail. `--overlap` (default `240`) makes adjacent windows smoother.

## Convert WAV or MP3 files to 16 kHz mono

Drag a WAV/MP3 file or a folder of WAV/MP3 files onto `run_resample_16k.bat`.

- A single file creates `audio-name_16k.wav` next to the original.
- A folder converts every WAV and MP3 within that folder and its subfolders, preserving the folder structure in a new sibling folder named `<folder>_16k`.

The conversion produces mono, 16-bit PCM WAV files at 16,000 Hz. The same `.venv` setup used by the spectrogram launcher is reused.
