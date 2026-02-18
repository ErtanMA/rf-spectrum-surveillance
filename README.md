# RF Spectrum Surveillance Monitor
### Project 1 — Crowdsourced Wideband Spectrum Surveillance

Implements a complete regulator-style spectrum monitoring pipeline. 
Currently tested on synthetic data.
ElectroSense PSD dataset (24 MHz – 1.7 GHz) will be used for verification.
---

![Spectrum Waterfall](assets/band_waterfall.png)

## Project structure

```
rf_spectrum_surveillance/
├── run_surveillance.py     - Main CLI 
├── download_data.py        - Dataset downloader
├── requirements.txt        - Requirements of the app versions needed
│
├── src/
│   ├── spectrum_monitor.py - DSP part of the code
│   ├── visualize.py        - Interactive plots
│   └── outputs.py          - CSV / JSON writers
│
├── notebooks/
│   └── report_notebook.ipynb  - Reproducible analysis (step-by-step)
│
├── data/                   - Downloaded Datasets should be here
└── outputs/                - All results will written here
```

---

## Setup

### 1. Prerequisites
- Python 3.10 or newer
- pip

### 2. Create a virtual environment (recommended)

```bash
# Navigate to the project folder
cd rf_spectrum_surveillance

# Create a virtual environment
python -m venv venv

# Activate it
# On macOS / Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Quick start — Demo mode (no dataset needed)

Run the full pipeline on synthetic data to verify everything works:

```bash
python run_surveillance.py --demo
```

This generates a synthetic PSD with injected emitters, runs all processing
steps, and writes output files to `outputs/`.

Open `outputs/band_waterfall.html` in your browser to see the interactive plot.

---

## Using the real ElectroSense dataset (No Full Test Done yet)

### Step 1 — List available files

```bash
python download_data.py --list
```

### Step 2 — Download file, starting from index 0.

```bash
python download_data.py --download 0
```

Files are saved to `data/` and converted to `.npz` automatically.

### Step 3 — Run the pipeline

```bash
python run_surveillance.py --input data/<filename>.npz --sensor my_sensor
```

---

## All CLI options

```
python run_surveillance.py --help

  --input FILE         Path to .npz / .npy / .csv PSD file
  --demo               Use synthetic demo data
  --sensor NAME        Sensor ID label                     [sensor_001]

  --noise-window N     Sliding window for noise floor      [60 frames]
  --noise-method       mad or quantile                     [mad]

  --pfa FLOAT          CA-CFAR false alarm probability     [0.001]
  --guard N            CFAR guard cells each side          [4]
  --training N         CFAR training cells each side       [32]
  --min-persist N      Required consecutive frame hits     [2]

  --min-area N         Minimum connected-component size    [2 bins]
  --outdir DIR         Output directory                    [outputs/]
  --no-plots           Skip HTML plot generation
```

### Examples

```bash
# Tighter false alarm control (fewer, more certain detections):
python run_surveillance.py --demo --pfa 0.0001

# Faster run without plots:
python run_surveillance.py --demo --no-plots

# Custom output directory:
python run_surveillance.py --demo --outdir results/run_01
```

---

## Output files

| File | Description |
|------|-------------|
| `outputs/events.csv` | One row per detected event: time, frequency, BW, power, tags |
| `outputs/sensor_summary.json` | Per-sensor occupancy stats and anomaly counts |
| `outputs/band_waterfall.html` | Interactive waterfall + event overlay (open in browser) |
| `outputs/noise_floor.html` | Mean PSD vs estimated noise floor |
| `outputs/event_features.html` | Bandwidth vs peak power scatter, colour by tag |

---

> **Jupyter Notebook** — Will be added (TBD)  
> The notebook will walk the user through every step interactively.

## Pipeline overview

```
PSD matrix (time × freq, dB)
        │
        ▼
Noise floor estimation
  · MAD sliding window per frequency bin
  · Robust to intermittent transmissions
        │
        ▼
CA-CFAR detection (per time frame)
  · Cell-Averaging CFAR along frequency axis
  · Guard cells: 4  |  Training cells: 32
  · Temporal persistence filter (≥2 consecutive frames)
        │
        ▼
2-D connected-component grouping
  · SciPy ndimage.label on binary detection map
  · Feature extraction per event:
      t_start/stop, f_centre, bandwidth, peak/mean power,
      integrated power, spectral flatness, kurtosis, edge slopes
        │
        ▼
Event classification
  · narrowband (<200 kHz) / wideband / broadband (>5 MHz)
  · burst (<5 s) / persistent (>300 s)
  · edge_violation (near licensed band edges)
        │
        ▼
Outputs: events.csv + sensor_summary.json + interactive plots
```

---

## Relevant papers and links

- ElectroSense PSD dataset (Zenodo): https://zenodo.org/records/7521246
- ElectroSense GitHub: https://github.com/electrosense/PSD-technology-classification-framework
- ElectroSense API: https://electrosense.networks.imdea.org/api-spec
