"""
RF Spectrum Surveillance Monitor
Project 1: Crowdsourced Wideband Spectrum Surveillance
Based on ElectroSense PSD dataset (Zenodo: https://zenodo.org/records/7521246)

Pipeline:
  PSD data → Noise floor estimation → CFAR detection → Event grouping → Feature extraction → Outputs
"""

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import median_abs_deviation
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_psd_data(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PSD matrix from a numpy .npz file (ElectroSense format).

    Expected keys in .npz:
        psd        : 2D array (n_time, n_freq) in dB
        frequencies: 1D array of frequency bin centres in Hz
        timestamps : 1D array of UNIX timestamps in seconds

    Returns
    -------
    psd         : np.ndarray  (n_time × n_freq), dB
    frequencies : np.ndarray  Hz
    timestamps  : np.ndarray  seconds
    """
    filepath = Path(filepath)
    ext = filepath.suffix.lower()

    if ext == ".npz":
        data = np.load(filepath, allow_pickle=True)
        psd = data["psd"].astype(np.float32)
        frequencies = data["frequencies"].astype(np.float64)
        timestamps = data["timestamps"].astype(np.float64)

    elif ext == ".npy":
        # Plain matrix – caller must supply freq/time axes separately
        psd = np.load(filepath).astype(np.float32)
        n_time, n_freq = psd.shape
        frequencies = np.arange(n_freq, dtype=np.float64)
        timestamps = np.arange(n_time, dtype=np.float64)

    elif ext == ".csv":
        df = pd.read_csv(filepath, index_col=0)
        psd = df.values.astype(np.float32)
        try:
            frequencies = np.array(df.columns, dtype=np.float64)
        except ValueError:
            frequencies = np.arange(psd.shape[1], dtype=np.float64)
        timestamps = np.arange(psd.shape[0], dtype=np.float64)

    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .npz, .npy, .csv")

    print(f"[load]  PSD shape: {psd.shape}  "
          f"| freq range: {frequencies[0]/1e6:.1f}–{frequencies[-1]/1e6:.1f} MHz  "
          f"| time steps: {len(timestamps)}")
    return psd, frequencies, timestamps


def generate_demo_psd(n_time=720, n_freq=512,
                      freq_start=24e6, freq_end=1700e6,
                      seed=42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic PSD waterfall for testing when no real dataset is available.
    Injects realistic-looking narrowband, broadband, and chirp-like emitters.
    """
    rng = np.random.default_rng(seed)
    frequencies = np.linspace(freq_start, freq_end, n_freq)
    timestamps = np.arange(n_time, dtype=np.float64) * 30.0   # 30-second steps → 6 hours

    # Noise floor: -90 dBm baseline with slow spatial/temporal drift
    noise = rng.normal(-90.0, 2.0, (n_time, n_freq)).astype(np.float32)

    psd = noise.copy()

    # --- Persistent narrowband carriers (licensed bands) ---
    for f_centre, bw_bins, power in [
        (100e6,  3,  -60),   # FM radio
        (450e6,  2,  -65),   # PMR446
        (900e6,  8,  -55),   # GSM
        (1800e6, 8,  -57),   # GSM-1800 (clipped to range)
    ]:
        if f_centre > freq_end:
            continue
        idx = np.argmin(np.abs(frequencies - f_centre))
        lo, hi = max(0, idx - bw_bins//2), min(n_freq, idx + bw_bins//2 + 1)
        psd[:, lo:hi] += power - (-90)   # relative to noise floor

    # --- Intermittent broadband interferer (e.g. switching PSU) ---
    start_t, dur_t = 100, 80
    lo_f, hi_f = 200, 230
    psd[start_t:start_t+dur_t, lo_f:hi_f] += rng.normal(25, 3,
                                                          (dur_t, hi_f-lo_f))

    # --- Sporadic short bursts (IoT / LPWAN) ---
    for _ in range(40):
        t0 = rng.integers(0, n_time - 5)
        f0 = rng.integers(10, n_freq - 10)
        psd[t0:t0+rng.integers(2, 6), f0:f0+rng.integers(1, 4)] += rng.uniform(15, 30)

    # --- Out-of-band emission near band edge ---
    guard_f = n_freq - 20
    psd[300:340, guard_f:guard_f+5] += 18.0

    print(f"[demo]  Synthetic PSD generated: {psd.shape}  "
          f"| freq {freq_start/1e6:.0f}–{freq_end/1e6:.0f} MHz")
    return psd, frequencies, timestamps


# ─────────────────────────────────────────────
# 2. NOISE FLOOR ESTIMATION
# ─────────────────────────────────────────────

def estimate_noise_floor(psd: np.ndarray,
                         window: int = 60,
                         method: str = "mad") -> tuple[np.ndarray, np.ndarray]:
    """
    Robust per-frequency-bin noise floor estimation using a sliding time window.

    Parameters
    ----------
    psd    : (n_time, n_freq)  input PSD in dB
    window : sliding window size in time frames
    method : "mad"      → median ± 1.4826 * MAD  (robust to interferers)
             "quantile" → low quantile baseline   (explicit false-alarm control)

    Returns
    -------
    mu    : (n_time, n_freq)  estimated noise floor (dB)
    sigma : (n_time, n_freq)  estimated spread (dB)
    """
    n_time, n_freq = psd.shape
    mu = np.empty_like(psd)
    sigma = np.empty_like(psd)

    half_w = window // 2

    for t in range(n_time):
        t_lo = max(0, t - half_w)
        t_hi = min(n_time, t + half_w + 1)
        segment = psd[t_lo:t_hi, :]   # (win, n_freq)

        if method == "mad":
            med = np.median(segment, axis=0)
            mad = median_abs_deviation(segment, axis=0)
            mu[t] = med
            sigma[t] = 1.4826 * mad + 1e-6     # avoid zero sigma

        elif method == "quantile":
            mu[t] = np.quantile(segment, 0.20, axis=0)   # 20th percentile
            sigma[t] = np.quantile(segment, 0.80, axis=0) - mu[t] + 1e-6

        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'mad' or 'quantile'.")

    return mu, sigma


# ─────────────────────────────────────────────
# 3. CFAR DETECTION
# ─────────────────────────────────────────────

def ca_cfar_1d(spectrum: np.ndarray,
               guard: int = 4,
               training: int = 32,
               pfa: float = 1e-3) -> np.ndarray:
    """
    1-D Cell-Averaging CFAR detector applied along frequency axis.

    Parameters
    ----------
    spectrum : 1D PSD row in dB
    guard    : guard cells each side (skip these in noise estimate)
    training : training cells each side
    pfa      : desired probability of false alarm (sets threshold multiplier)

    Returns
    -------
    detections : boolean array, True where target detected
    """
    n = len(spectrum)
    detections = np.zeros(n, dtype=bool)

    # Threshold factor α for CA-CFAR: α = N * (PFA^(-1/N) - 1)
    N = 2 * training
    alpha = N * (pfa ** (-1.0 / N) - 1.0)

    for i in range(n):
        lo_train = max(0, i - guard - training)
        lo_guard = max(0, i - guard)
        hi_guard = min(n, i + guard + 1)
        hi_train = min(n, i + guard + training + 1)

        cells = np.concatenate([spectrum[lo_train:lo_guard],
                                 spectrum[hi_guard:hi_train]])
        if len(cells) < 4:
            continue

        noise_est = np.mean(cells)
        threshold = noise_est + alpha   # dB domain: additive threshold

        detections[i] = spectrum[i] > threshold

    return detections


def apply_cfar_full(psd: np.ndarray,
                    mu: np.ndarray,
                    guard: int = 4,
                    training: int = 32,
                    pfa: float = 1e-3,
                    min_persistence: int = 2) -> np.ndarray:
    """
    Apply CA-CFAR across every time frame of the PSD, then enforce temporal
    persistence (event must appear in ≥ min_persistence consecutive frames).

    Returns
    -------
    detection_map : boolean (n_time, n_freq)
    """
    n_time, n_freq = psd.shape
    raw_detections = np.zeros((n_time, n_freq), dtype=bool)

    # Subtract noise floor so CFAR sees a whitened spectrum
    whitened = psd - mu

    for t in range(n_time):
        raw_detections[t] = ca_cfar_1d(whitened[t], guard=guard,
                                        training=training, pfa=pfa)

    # Temporal persistence filter: require ≥ min_persistence consecutive hits
    if min_persistence > 1:
        kernel = np.ones((min_persistence, 1), dtype=np.float32)
        count = ndimage.convolve(raw_detections.astype(np.float32),
                                 kernel, mode="constant", cval=0)
        detection_map = count >= min_persistence
    else:
        detection_map = raw_detections

    n_events = detection_map.sum()
    print(f"[cfar]  Detections after persistence filter: {n_events:,} bins  "
          f"({100*n_events/detection_map.size:.2f}% of map)")
    return detection_map


# ─────────────────────────────────────────────
# 4. EVENT GROUPING (2-D CONNECTED COMPONENTS)
# ─────────────────────────────────────────────

def group_events(detection_map: np.ndarray,
                 psd: np.ndarray,
                 frequencies: np.ndarray,
                 timestamps: np.ndarray,
                 min_area: int = 2) -> list[dict]:
    """
    Form 2-D connected components in the time-frequency detection map and
    extract interpretable features for each event.

    Returns
    -------
    events : list of dicts, one per event
    """
    labeled, n_labels = ndimage.label(detection_map)
    print(f"[group] Connected components: {n_labels}")

    freq_step = (frequencies[-1] - frequencies[0]) / (len(frequencies) - 1)

    events = []
    for label_id in range(1, n_labels + 1):
        mask = labeled == label_id
        area = mask.sum()
        if area < min_area:
            continue

        t_idx, f_idx = np.where(mask)
        power_vals = psd[t_idx, f_idx]

        t_start = timestamps[t_idx.min()]
        t_stop  = timestamps[t_idx.max()]
        f_lo    = frequencies[f_idx.min()]
        f_hi    = frequencies[f_idx.max()]
        f0      = (f_lo + f_hi) / 2.0
        bw      = (f_hi - f_lo) + freq_step

        peak_power = float(power_vals.max())
        mean_power = float(power_vals.mean())

        # Convert dB → linear for integrated power
        power_lin = 10 ** (power_vals / 10.0)
        integrated_dbhz = float(10 * np.log10(power_lin.sum() * freq_step + 1e-30))

        # Spectral shape of the event (1-D PSD slice at peak time)
        peak_t = t_idx[np.argmax(power_vals)]
        event_slice = psd[peak_t, f_idx.min():f_idx.max()+1]
        spectral_flatness = _spectral_flatness(event_slice)
        spectral_kurtosis = float(np.nan_to_num(
            (np.mean((event_slice - event_slice.mean())**4)) /
            (np.mean((event_slice - event_slice.mean())**2)**2 + 1e-10) - 3
        ))

        # Edge slope (dB / MHz)
        if len(event_slice) >= 4:
            slope_lo = float((event_slice[1] - event_slice[0]) / (freq_step / 1e6 + 1e-9))
            slope_hi = float((event_slice[-1] - event_slice[-2]) / (freq_step / 1e6 + 1e-9))
        else:
            slope_lo = slope_hi = 0.0

        event = dict(
            event_id=label_id,
            t_start=float(t_start),
            t_stop=float(t_stop),
            duration_s=float(t_stop - t_start),
            f_centre_hz=float(f0),
            f_centre_mhz=float(f0 / 1e6),
            bandwidth_hz=float(bw),
            bandwidth_khz=float(bw / 1e3),
            peak_power_db=peak_power,
            mean_power_db=mean_power,
            integrated_power_dbhz=integrated_dbhz,
            spectral_flatness=float(spectral_flatness),
            spectral_kurtosis=spectral_kurtosis,
            slope_lo_db_per_mhz=slope_lo,
            slope_hi_db_per_mhz=slope_hi,
            area_bins=int(area),
        )
        events.append(event)

    print(f"[group] Events after min_area filter: {len(events)}")
    return events


def _spectral_flatness(spectrum_db: np.ndarray) -> float:
    """Spectral flatness (Wiener entropy) in linear domain."""
    if len(spectrum_db) < 2:
        return 0.0
    lin = 10 ** (spectrum_db / 10.0)
    lin = np.clip(lin, 1e-30, None)
    geo = np.exp(np.mean(np.log(lin)))
    ari = np.mean(lin)
    return float(geo / (ari + 1e-30))


# ─────────────────────────────────────────────
# 5. EVENT CLASSIFICATION / TAGGING
# ─────────────────────────────────────────────

def classify_events(events: list[dict],
                    guard_band_margin_hz: float = 200e3,
                    licensed_band_edges: list[tuple] | None = None) -> list[dict]:
    """
    Assign regulator-style tags to each event.

    Tags
    ----
    narrowband   : bandwidth < 200 kHz
    broadband    : bandwidth > 5 MHz
    burst        : duration < 5 seconds
    persistent   : duration > 300 seconds
    edge_violation: centre frequency within guard_band_margin_hz of a licensed edge

    Parameters
    ----------
    licensed_band_edges : list of (lower_edge_hz, upper_edge_hz) tuples
                          e.g. [(87.5e6, 108e6), (880e6, 960e6)]
    """
    if licensed_band_edges is None:
        # Common European band edges (illustrative)
        licensed_band_edges = [
            (87.5e6,  108e6),    # FM broadcast
            (174e6,   230e6),    # DAB / DVB-T
            (470e6,   790e6),    # UHF TV
            (880e6,   960e6),    # GSM-900
            (1710e6, 1880e6),    # GSM-1800
        ]

    for ev in events:
        tags = []
        bw = ev["bandwidth_hz"]
        dur = ev["duration_s"]
        fc = ev["f_centre_hz"]

        if bw < 200e3:
            tags.append("narrowband")
        elif bw > 5e6:
            tags.append("broadband")
        else:
            tags.append("wideband")

        if dur < 5:
            tags.append("burst")
        elif dur > 300:
            tags.append("persistent")

        # Out-of-band / guard-band check
        for lo, hi in licensed_band_edges:
            if abs(fc - lo) < guard_band_margin_hz or abs(fc - hi) < guard_band_margin_hz:
                tags.append("edge_violation")
                break

        ev["tags"] = ",".join(tags)

    return events


# ─────────────────────────────────────────────
# 6. OCCUPANCY STATISTICS
# ─────────────────────────────────────────────

def compute_occupancy(psd: np.ndarray,
                      mu: np.ndarray,
                      sigma: np.ndarray,
                      threshold_sigma: float = 3.0) -> np.ndarray:
    """
    Per-frequency-bin occupancy: fraction of time frames where signal exceeds
    noise floor + threshold_sigma * sigma.

    Returns
    -------
    occupancy : 1D array (n_freq,) in range [0, 1]
    """
    above = psd > (mu + threshold_sigma * sigma)
    return above.mean(axis=0)


# ─────────────────────────────────────────────
# 7. WELCH PSD VALIDATION MODULE  (IQ → PSD)
# ─────────────────────────────────────────────

def welch_psd_from_iq(iq_samples: np.ndarray,
                      fs: float,
                      nfft: int = 4096,
                      overlap: float = 0.75,
                      n_segments: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a Welch PSD from raw complex IQ samples using Hann window.
    Used to validate that the detector works identically on PSD derived from IQ.

    Parameters
    ----------
    iq_samples : 1D complex64/128 array
    fs         : sample rate in Hz
    nfft       : FFT length (bin width = fs / nfft)
    overlap    : fractional overlap between segments (0–1)
    n_segments : number of segments to average

    Returns
    -------
    freqs : 1D Hz
    psd   : 1D dBm/Hz (relative power spectral density)
    """
    from scipy.signal import welch
    hop = int(nfft * (1 - overlap))
    f, Pxx = welch(iq_samples, fs=fs, window="hann", nperseg=nfft,
                   noverlap=nfft - hop, scaling="density")
    psd_db = 10 * np.log10(np.abs(Pxx) + 1e-30)
    return f, psd_db
