"""
Visualization module: interactive Plotly waterfall + event overlays,
occupancy plots, and event timeline.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path


def plot_waterfall(psd: np.ndarray,
                   frequencies: np.ndarray,
                   timestamps: np.ndarray,
                   events: list[dict] | None = None,
                   detection_map: np.ndarray | None = None,
                   title: str = "Spectrum Waterfall",
                   output_path: str | None = None,
                   max_freq_bins: int = 512,
                   max_time_bins: int = 512) -> go.Figure:
    """
    Build an interactive Plotly waterfall (heatmap) with optional event overlays.

    Downsamples display to max_freq_bins × max_time_bins for browser performance.
    """
    n_time, n_freq = psd.shape

    # Downsample for rendering
    t_step = max(1, n_time // max_time_bins)
    f_step = max(1, n_freq // max_freq_bins)
    psd_ds = psd[::t_step, ::f_step]
    freq_ds = frequencies[::f_step] / 1e6        # → MHz
    time_ds = timestamps[::t_step]

    # Convert timestamps to readable labels
    from datetime import datetime
    try:
        time_labels = [datetime.utcfromtimestamp(ts).strftime("%H:%M:%S")
                       for ts in time_ds]
    except (OSError, ValueError):
        time_labels = [f"{int(ts)}s" for ts in time_ds]

    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.60, 0.20, 0.20],
        shared_xaxes=True,
        subplot_titles=("Waterfall (PSD dB)", "Occupancy (%)", "Event Timeline"),
        vertical_spacing=0.07
    )

    # ── Row 1: Waterfall heatmap ──────────────────────────────────────────────
    vmin = float(np.percentile(psd_ds, 2))
    vmax = float(np.percentile(psd_ds, 98))

    fig.add_trace(go.Heatmap(
        z=psd_ds,
        x=freq_ds,
        y=time_labels,
        colorscale="Viridis",
        zmin=vmin,
        zmax=vmax,
        colorbar=dict(title="dB", len=0.55, y=0.72),
        hovertemplate="Freq: %{x:.2f} MHz<br>Time: %{y}<br>Power: %{z:.1f} dB<extra></extra>",
        name="PSD",
    ), row=1, col=1)

    # ── Event overlays on waterfall ──────────────────────────────────────────
    if events:
        # Top-20 events by peak power, limit overlay clutter
        top_events = sorted(events, key=lambda e: e["peak_power_db"], reverse=True)[:40]

        ev_x, ev_y, ev_text, ev_color = [], [], [], []
        color_map = {"narrowband": "#ff4444", "broadband": "#ffaa00",
                     "wideband": "#44aaff", "edge_violation": "#ff00ff"}

        for ev in top_events:
            fc_mhz = ev["f_centre_mhz"]
            # Find nearest time label index
            t_idx = np.argmin(np.abs(timestamps - ev["t_start"]))
            t_ds_idx = t_idx // t_step
            t_label = time_labels[min(t_ds_idx, len(time_labels)-1)]

            first_tag = ev["tags"].split(",")[0] if ev["tags"] else "wideband"
            col = color_map.get(first_tag, "#ffffff")

            ev_x.append(fc_mhz)
            ev_y.append(t_label)
            ev_text.append(
                f"ID:{ev['event_id']}<br>"
                f"f0={fc_mhz:.2f} MHz<br>"
                f"BW={ev['bandwidth_khz']:.1f} kHz<br>"
                f"Peak={ev['peak_power_db']:.1f} dB<br>"
                f"Tags: {ev['tags']}"
            )
            ev_color.append(col)

        fig.add_trace(go.Scatter(
            x=ev_x, y=ev_y,
            mode="markers",
            marker=dict(symbol="x", size=10, color=ev_color,
                        line=dict(width=2, color=ev_color)),
            text=ev_text,
            hoverinfo="text",
            name="Detected Events",
        ), row=1, col=1)

    # ── Row 2: Occupancy bar ─────────────────────────────────────────────────
    occupancy_ds = np.zeros(len(freq_ds))
    if detection_map is not None:
        occ_full = detection_map.mean(axis=0)
        occupancy_ds = occ_full[::f_step] * 100   # percent

    fig.add_trace(go.Bar(
        x=freq_ds, y=occupancy_ds,
        name="Occupancy (%)",
        marker_color="#44ccff",
        hovertemplate="Freq: %{x:.2f} MHz<br>Occupancy: %{y:.1f}%<extra></extra>",
    ), row=2, col=1)

    # ── Row 3: Event timeline (scatter: time vs f_centre) ───────────────────
    if events:
        ev_df = pd.DataFrame(events)
        bw_norm = (ev_df["bandwidth_hz"] - ev_df["bandwidth_hz"].min()) / \
                  (ev_df["bandwidth_hz"].max() - ev_df["bandwidth_hz"].min() + 1)
        marker_sizes = (bw_norm * 16 + 5).clip(5, 20)

        try:
            from datetime import datetime
            ev_times = [datetime.utcfromtimestamp(ts).strftime("%H:%M:%S")
                        for ts in ev_df["t_start"]]
        except Exception:
            ev_times = ev_df["t_start"].astype(str).tolist()

        fig.add_trace(go.Scatter(
            x=ev_df["f_centre_mhz"],
            y=ev_times,
            mode="markers",
            marker=dict(
                size=marker_sizes,
                color=ev_df["peak_power_db"],
                colorscale="Hot",
                showscale=True,
                colorbar=dict(title="Peak dB", len=0.20, y=0.08),
            ),
            text=ev_df.apply(lambda r: f"ID:{r.event_id} {r.tags}", axis=1),
            hoverinfo="text",
            name="Events",
        ), row=3, col=1)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        height=900,
        template="plotly_dark",
        showlegend=True,
        legend=dict(x=1.05, y=1),
        margin=dict(l=60, r=120, t=80, b=50),
    )
    fig.update_xaxes(title_text="Frequency (MHz)", row=3, col=1)
    fig.update_xaxes(title_text="Frequency (MHz)", row=2, col=1)
    fig.update_yaxes(title_text="Time (UTC)", row=1, col=1)
    fig.update_yaxes(title_text="Occupancy %", row=2, col=1)
    fig.update_yaxes(title_text="Time", row=3, col=1)

    if output_path:
        fig.write_html(output_path)
        print(f"[viz]   Waterfall saved → {output_path}")

    return fig


def plot_noise_floor(psd: np.ndarray,
                     mu: np.ndarray,
                     frequencies: np.ndarray,
                     output_path: str | None = None) -> go.Figure:
    """
    Plot measured PSD (time-average) vs estimated noise floor.
    """
    freq_mhz = frequencies / 1e6
    avg_psd = psd.mean(axis=0)
    avg_mu = mu.mean(axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freq_mhz, y=avg_psd, mode="lines",
                             name="Mean PSD", line=dict(color="#44aaff", width=1)))
    fig.add_trace(go.Scatter(x=freq_mhz, y=avg_mu, mode="lines",
                             name="Noise Floor (est.)", line=dict(color="#ff8800",
                             width=2, dash="dash")))
    fig.update_layout(
        title="Mean PSD vs Estimated Noise Floor",
        xaxis_title="Frequency (MHz)", yaxis_title="Power (dB)",
        template="plotly_dark", height=400,
    )

    if output_path:
        fig.write_html(output_path)
        print(f"[viz]   Noise floor plot saved → {output_path}")
    return fig


def plot_event_features(events: list[dict],
                        output_path: str | None = None) -> go.Figure:
    """
    2-D scatter of bandwidth vs peak power, colour-coded by tag.
    Hover shows full feature vector.
    """
    if not events:
        return go.Figure()

    df = pd.DataFrame(events)

    color_seq = px.colors.qualitative.Bold
    tag_list = sorted({t for tags in df["tags"] for t in tags.split(",") if t})
    tag_color = {tag: color_seq[i % len(color_seq)] for i, tag in enumerate(tag_list)}

    fig = go.Figure()
    for tag in tag_list:
        mask = df["tags"].str.contains(tag)
        sub = df[mask]
        if sub.empty:
            continue
        hover = sub.apply(
            lambda r: (f"ID:{r.event_id}<br>f0={r.f_centre_mhz:.2f} MHz<br>"
                       f"BW={r.bandwidth_khz:.1f} kHz<br>Peak={r.peak_power_db:.1f} dB<br>"
                       f"Duration={r.duration_s:.1f} s<br>Tags:{r.tags}"), axis=1
        )
        fig.add_trace(go.Scatter(
            x=sub["bandwidth_khz"], y=sub["peak_power_db"],
            mode="markers",
            marker=dict(size=8, color=tag_color[tag], opacity=0.8),
            text=hover, hoverinfo="text",
            name=tag,
        ))

    fig.update_layout(
        title="Event Feature Space: Bandwidth vs Peak Power",
        xaxis_title="Bandwidth (kHz, log)", yaxis_title="Peak Power (dB)",
        xaxis_type="log", template="plotly_dark", height=500,
    )
    if output_path:
        fig.write_html(output_path)
        print(f"[viz]   Feature plot saved → {output_path}")
    return fig
