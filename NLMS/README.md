# SIH26052 — Two-Channel NLMS Adaptive Noise Cancellation

This is an offline, time-domain implementation of two-channel normalized least-mean-squares (NLMS) adaptive noise cancellation. It deliberately does not use STFT/iSTFT, AI models, or live microphone input.

```text
primary.wav (speech + noise) ─┐
                              ├─ manual NLMS ─ estimated noise ─┐
reference.wav (related noise)─┘                                  ├─ primary − estimate → enhanced speech
                                                                   │
```

## Included

- A manual NumPy sample-by-sample NLMS loop with configurable filter length, `mu`, and `epsilon`.
- In-memory preprocessing only: stereo-to-mono conversion, 16 kHz resampling, DC removal, common-length trimming, and optional small-delay correction.
- Required outputs: `outputs/estimated_noise.wav` and `outputs/nlms_enhanced.wav`.
- Debug artifacts: final coefficients and run metadata.
- Required waveform and spectrogram figures.
- Input/output SNR, SNR improvement, SI-SDR, optional STOI, and processing-time evaluation.
- Reproducible controlled and separately seeded held-out WAV tests, plus an automated parameter sweep.

## Setup

Use Python 3.10+:

```powershell
python -m pip install -r requirements.txt
```

STOI is optional:

```powershell
python -m pip install pystoi
```

## Input contract

Place the microphone pair in:

```text
data/
├── primary.wav       # target speech + noise
└── reference.wav     # correlated noise, ideally little/no target speech
```

Inputs may be multichannel and need not already be 16 kHz. The program prepares copies in memory and never changes the original files. It converts each channel to mono, resamples to 16 kHz, removes DC, and trims both signals to the shared length. The default delay estimator searches ±30 ms, zero-pads the reference shift, and can be disabled when microphones are already aligned. If `primary.wav` and `reference.wav` are placed in the project root, those explicit files take priority over the `data` pair.

The two recordings must contain the same noise event at the same time. The program now measures their peak normalized correlation before NLMS and rejects a weakly related pair by default, because adaptive cancellation cannot work from an unrelated reference. If this guard triggers, provide synchronized recordings rather than forcing the run.

## Run

```powershell
python nlms.py
```

This creates:

```text
outputs/
├── estimated_noise.wav
├── nlms_enhanced.wav
├── final_filter_coefficients.npy
├── nlms_metadata.json
└── plots/
    ├── waveforms.png
    └── spectrograms.png
```

The waveform plot shows primary, reference, estimated noise, and enhanced output. The spectrogram plot shows primary, estimated noise, and enhanced output. WAVs are written as float WAV without output normalization or clipping, so they directly reflect the NLMS calculation.

Tune without editing the core implementation:

```powershell
python nlms.py --filter-length 64 --mu 0.05 --epsilon 1e-8
python nlms.py --no-delay-correction
python nlms.py --max-delay-ms 15
```

For diagnosis only, an unsafe run can be forced:

```powershell
python nlms.py --allow-low-correlation
```

Use `mu` in `(0, 2]`. The conservative default is `0.05`; values around `0.03–0.1` are good starting points when speech is present in the primary channel. Longer filters model longer acoustic paths but converge more slowly. Keep `epsilon` small and positive.

To run a recording pair outside `data`:

```powershell
python nlms.py --primary D:\recordings\primary.wav --reference D:\recordings\reference.wav --output-dir outputs\field_run
```

## Evaluation

With a clean-speech reference aligned to the primary channel:

```powershell
python nlms.py --clean data\clean_speech.wav

# Or evaluate a completed run:
python evaluate.py --clean data\clean_speech.wav
```

Metrics are calculated from the actual WAVs; the project does not invent performance values. STOI is reported as unavailable unless `pystoi` is installed.

## Controlled and held-out tests

Generated WAV files are intentionally excluded from source control. Create a controlled case with known clean speech-like content and known primary noise, plus a separately seeded held-out case:

```powershell
python generate_test_data.py
python nlms.py --primary data\primary.wav --reference data\reference.wav --clean data\clean_speech.wav
python evaluate.py --primary data\primary.wav --clean data\clean_speech.wav

# Independent held-out input, not used by the unit test:
python nlms.py --primary data\unseen\primary.wav --reference data\unseen\reference.wav --output-dir outputs\unseen --clean data\unseen\clean_speech.wav
```

The controlled source is a clearly labelled speech-like surrogate for repeatability, not a claim of human speech. It saves `known_primary_noise.wav` so the noise estimate can be compared directly. Run the same pipeline on an externally captured, unseen synchronized microphone pair for field validation.

Run automated verification:

```powershell
python -m unittest discover -s tests -v
```

The test verifies that the estimated noise correlates with known primary noise, output SNR improves, and several filter-length/`mu` configurations remain finite. Sweep a broader range of noise levels and settings:

```powershell
python run_sweep.py
```

This writes `outputs/parameter_sweep.csv` with observed results for three input noise levels, three filter lengths, and three `mu` values.

## NLMS algorithm

For every sample `n`, the filter stores the most-recent-first reference vector `x(n)` and coefficient vector `w(n)`:

```text
estimated_noise(n) = w(n)^T x(n)
enhanced(n)        = primary(n) − estimated_noise(n)
w(n+1)             = w(n) + mu * enhanced(n) * x(n) /
                           (epsilon + ||x(n)||²)
```

Only reference-correlated noise can be cancelled. Noise uncorrelated between microphones will remain, and target-speech leakage into the reference can suppress desired speech. Good microphone synchronization, a reference near the noise source, and low target-speech leakage are important for reliable results.
