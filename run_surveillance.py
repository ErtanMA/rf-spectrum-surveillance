#!/usr/bin/env python3
"""
run_surveillance.py  –  RF Spectrum Surveillance CLI
=====================================================

Usage
-----
# Demo mode (no dataset needed – generates synthetic data):
    python run_surveillance.py --demo

# Real ElectroSense PSD dataset:
    python run_surveillance.py --input data/psd_data.npz --sensor my_sensor_01

# Quick run with custom thresholds:
    python run_surveillance.py --demo --pfa 0.001 --window 30 --min-persist 2

Full options:
    python run_surveillance.py --help
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from spectrum_monitor import (
    load_psd_data,
    generate_demo_psd,
    estimate_noise_floor,
    apply_cfar_full,
    group_events,
    classify_events,
    compute_occupancy,
)
from visualize import plot_waterfall, plot_noise_floor, plot_event_features
from outputs import save_events_csv, save_sensor_summary


def parse_args():
    p = argparse.ArgumentParser(
        description="RF Spectrum Surveillance Monitor (ElectroSense PSD pipeline)"
    )
    # Input
    p.add_argument("--input", type=str, default=None,
                   help="Path to PSD data file (.npz / .npy / .csv). "
                        "Use --demo to generate synthetic data instead.")
    p.add_argument("--demo", action="store_true",
                   help="Run on synthetic demo data (no dataset required).")
    p.add_argument("--sensor", type=str, default="sensor_001",
                   help="Sensor ID label for output files.")

    # Noise floor
    p.add_argument("--noise-window", type=int, default=60,
                   help="Sliding window (time frames) for noise floor estimation. [60]")
    p.add_argument("--noise-method", choices=["mad", "quantile"], default="mad",
                   help="Noise floor estimation method: mad or quantile. [mad]")

    # CFAR
    p.add_argument("--pfa", type=float, default=1e-3,
                   help="CA-CFAR probability of false alarm. [0.001]")
    p.add_argument("--guard", type=int, default=4,
                   help="CA-CFAR guard cells each side. [4]")
    p.add_argument("--training", type=int, default=32,
                   help="CA-CFAR training cells each side. [32]")
    p.add_argument("--min-persist", type=int, default=2,
                   help="Minimum persistence (consecutive frames) for an event. [2]")

    # Event grouping
    p.add_argument("--min-area", type=int, default=2,
                   help="Minimum area (bins) for a connected component to be kept. [2]")

    # Output
    p.add_argument("--outdir", type=str, default="outputs",
                   help="Directory to write output files. [outputs/]")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip interactive HTML plot generation.")

    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load / generate data ─────────────────────────────────────────────
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  RF SPECTRUM SURVEILLANCE MONITOR")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    if args.demo:
        print("[step 1/7]  Generating synthetic demo PSD …")
        psd, frequencies, timestamps = generate_demo_psd()
    elif args.input:
        print(f"[step 1/7]  Loading PSD data from {args.input} …")
        psd, frequencies, timestamps = load_psd_data(args.input)
    else:
        print("ERROR: Provide --input <file> or --demo.\n")
        sys.exit(1)

    # ── 2. Noise floor estimation ───────────────────────────────────────────
    print(f"\n[step 2/7]  Estimating noise floor "
          f"(method={args.noise_method}, window={args.noise_window} frames) …")
    mu, sigma = estimate_noise_floor(psd,
                                     window=args.noise_window,
                                     method=args.noise_method)

    # ── 3. CFAR detection ───────────────────────────────────────────────────
    print(f"\n[step 3/7]  Running CA-CFAR detector "
          f"(PFA={args.pfa}, guard={args.guard}, training={args.training}) …")
    detection_map = apply_cfar_full(psd, mu,
                                    guard=args.guard,
                                    training=args.training,
                                    pfa=args.pfa,
                                    min_persistence=args.min_persist)

    # ── 4. Event grouping ───────────────────────────────────────────────────
    print(f"\n[step 4/7]  Grouping 2-D connected components (min_area={args.min_area}) …")
    events = group_events(detection_map, psd, frequencies, timestamps,
                          min_area=args.min_area)

    # ── 5. Event classification ─────────────────────────────────────────────
    print(f"\n[step 5/7]  Classifying and tagging events …")
    events = classify_events(events)

    tag_summary = {}
    for ev in events:
        for t in ev["tags"].split(","):
            if t:
                tag_summary[t] = tag_summary.get(t, 0) + 1
    for tag, count in sorted(tag_summary.items(), key=lambda x: -x[1]):
        print(f"           {tag:20s}: {count:4d} events")

    # ── 6. Occupancy statistics ─────────────────────────────────────────────
    print(f"\n[step 6/7]  Computing occupancy statistics …")
    occupancy = compute_occupancy(psd, mu, sigma, threshold_sigma=3.0)
    top5_idx = occupancy.argsort()[-5:][::-1]
    print("           Top-5 busiest frequency bins:")
    for idx in top5_idx:
        print(f"             {frequencies[idx]/1e6:8.2f} MHz  →  "
              f"{occupancy[idx]*100:.1f}% occupied")

    # ── 7. Outputs ──────────────────────────────────────────────────────────
    print(f"\n[step 7/7]  Writing outputs to {outdir}/ …")

    events_csv = outdir / "events.csv"
    summary_json = outdir / "sensor_summary.json"

    save_events_csv(events, str(events_csv))
    save_sensor_summary(events, psd, frequencies, timestamps, occupancy,
                        sensor_id=args.sensor,
                        output_path=str(summary_json))

    if not args.no_plots:
        waterfall_html = outdir / "band_waterfall.html"
        noise_html     = outdir / "noise_floor.html"
        features_html  = outdir / "event_features.html"

        plot_waterfall(psd, frequencies, timestamps,
                       events=events,
                       detection_map=detection_map,
                       title=f"Spectrum Waterfall – {args.sensor}",
                       output_path=str(waterfall_html))

        plot_noise_floor(psd, mu, frequencies,
                         output_path=str(noise_html))

        plot_event_features(events,
                            output_path=str(features_html))

    elapsed = time.time() - t0
    print(f"\n✔  Pipeline complete in {elapsed:.1f}s")
    print(f"   Events detected : {len(events)}")
    print(f"   Output dir      : {outdir.resolve()}\n")
    print("   Files written:")
    for f in sorted(outdir.iterdir()):
        print(f"     {f.name}")
    print()


if __name__ == "__main__":
    main()
