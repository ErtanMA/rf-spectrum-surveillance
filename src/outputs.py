"""
Output writers: events.csv, sensor_summary.json
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def save_events_csv(events: list[dict], output_path: str) -> pd.DataFrame:
    """
    Write detected events to CSV.  One row per event.

    Columns
    -------
    event_id, t_start, t_stop, duration_s,
    f_centre_hz, f_centre_mhz, bandwidth_hz, bandwidth_khz,
    peak_power_db, mean_power_db, integrated_power_dbhz,
    spectral_flatness, spectral_kurtosis,
    slope_lo_db_per_mhz, slope_hi_db_per_mhz,
    area_bins, tags
    """
    if not events:
        print("[out]   No events to save.")
        return pd.DataFrame()

    df = pd.DataFrame(events)

    # Round floats for readability
    float_cols = [c for c in df.columns if df[c].dtype == float]
    df[float_cols] = df[float_cols].round(4)

    df.to_csv(output_path, index=False)
    print(f"[out]   Events saved → {output_path}  ({len(df)} rows)")
    return df


def save_sensor_summary(events: list[dict],
                        psd: np.ndarray,
                        frequencies: np.ndarray,
                        timestamps: np.ndarray,
                        occupancy: np.ndarray,
                        sensor_id: str = "sensor_001",
                        output_path: str = "sensor_summary.json") -> dict:
    """
    Write per-sensor occupancy statistics and anomaly counts to JSON.
    """
    n_events = len(events)
    tag_counts: dict[str, int] = {}
    for ev in events:
        for tag in ev.get("tags", "").split(","):
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    duration_s = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0

    # Band occupancy summary (coarse bands)
    band_occ = {}
    band_defs = {
        "24_100_MHz": (24e6, 100e6),
        "100_300_MHz": (100e6, 300e6),
        "300_600_MHz": (300e6, 600e6),
        "600_1000_MHz": (600e6, 1000e6),
        "1000_1700_MHz": (1000e6, 1700e6),
    }
    for band_name, (flo, fhi) in band_defs.items():
        mask = (frequencies >= flo) & (frequencies <= fhi)
        if mask.sum() > 0:
            band_occ[band_name] = round(float(occupancy[mask].mean() * 100), 2)
        else:
            band_occ[band_name] = None

    # Noise floor stability (variance of estimated floor per band)
    mean_psd_db = float(psd.mean())
    std_psd_db  = float(psd.std())

    summary = {
        "sensor_id": sensor_id,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
        "recording_duration_s": round(duration_s, 1),
        "freq_range_hz": {
            "start": float(frequencies[0]),
            "stop": float(frequencies[-1]),
        },
        "n_time_frames": int(psd.shape[0]),
        "n_freq_bins": int(psd.shape[1]),
        "total_events_detected": n_events,
        "event_type_counts": tag_counts,
        "mean_psd_db": round(mean_psd_db, 2),
        "std_psd_db": round(std_psd_db, 2),
        "band_occupancy_percent": band_occ,
        "top_events_by_power": _top_events(events, key="peak_power_db", n=5),
        "top_events_by_bandwidth": _top_events(events, key="bandwidth_hz", n=5),
        "edge_violations": [ev for ev in events
                            if "edge_violation" in ev.get("tags", "")],
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_serial)

    print(f"[out]   Sensor summary saved → {output_path}")
    return summary


def _top_events(events: list[dict], key: str, n: int = 5) -> list[dict]:
    sorted_ev = sorted(events, key=lambda e: e.get(key, 0), reverse=True)[:n]
    return [{k: v for k, v in ev.items()
             if k in ("event_id", "f_centre_mhz", "bandwidth_khz",
                      "peak_power_db", "duration_s", "tags")}
            for ev in sorted_ev]


def _json_serial(obj):
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serialisable")
