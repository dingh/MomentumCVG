"""
Chain Tradability Diagnostic Plots (Plotly).

10 interactive plot functions for evaluating option chain tradability:

Single-ticker plots (accept a ChainSlice):
    1. plot_iv_surface        — IV heatmap (strike × DTE)
    2. plot_spread_heatmap    — Spread% heatmap (strike × DTE)
    3. plot_oi_volume         — OI & volume by strike (single expiry)
    4. plot_smile             — IV smile / skew (single expiry)
    5. plot_term_structure    — ATM IV vs DTE
    6. plot_theta_per_capital — Theta carry efficiency (single expiry)
    7. plot_straddle_friction — ATM straddle round-trip cost by DTE

Cross-sectional / time-series plots (accept pre-computed Series/DataFrames):
    8. plot_tradability_distribution — friction histogram across universe
    9. plot_tradability_timeseries   — friction & pass-rate over time
   10. plot_signal_tradability       — signal picks vs universe comparison

Helper functions:
    compute_atm_friction      — friction for one ChainSlice + expiry
    compute_universe_friction — batch friction across tickers on one date
"""

from __future__ import annotations

from datetime import date
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .chain_slice import ChainSlice, EPS


# ======================================================================
# Helper utilities
# ======================================================================

def _atm_row(df: pd.DataFrame) -> pd.Series:
    """Return the row closest to ATM (smallest |strike - stockPrice|)."""
    dist = (df["strike"] - df["stockPrice"]).abs()
    return df.loc[dist.idxmin()]


def compute_atm_friction(cs: ChainSlice, expiry_date: date) -> float:
    """
    Compute straddle friction at the ATM strike for one expiry.

    friction = (callSpr + putSpr) / max(callMid + putMid, eps)

    Returns NaN if no data for that expiry.
    """
    sub = cs.filter_expiry(expiry_date)
    if sub.empty:
        return float("nan")

    row = _atm_row(sub)
    denom = max(row["callMid"] + row["putMid"], EPS)
    return float((row["callSpr"] + row["putSpr"]) / denom)


def compute_universe_friction(
    loader,  # ChainLoader (avoid circular import)
    tickers: List[str],
    trade_date: date,
    target_dte: int = 7,
    dte_tolerance: int = 4,
) -> pd.Series:
    """
    Batch-compute ATM straddle friction for many tickers on one date.

    For each ticker, finds the expiry closest to *target_dte* (within
    *dte_tolerance*) and computes the ATM friction ratio.

    Returns a ``pd.Series`` indexed by ticker.
    """
    slices = loader.load_chain_multi(tickers, trade_date)
    result: Dict[str, float] = {}

    for tkr, cs in slices.items():
        # Find expiry closest to target_dte
        best_expiry = None
        best_diff = 999
        for exp in cs.available_expiries():
            dte = (exp - trade_date).days
            diff = abs(dte - target_dte)
            if dte > 0 and diff <= dte_tolerance and diff < best_diff:
                best_diff = diff
                best_expiry = exp

        if best_expiry is None:
            result[tkr] = float("nan")
            continue

        result[tkr] = compute_atm_friction(cs, best_expiry)

    return pd.Series(result, name="atm_friction")


# ======================================================================
# Plot 1 — IV Surface
# ======================================================================

def plot_iv_surface(
    cs: ChainSlice,
    iv_col: str = "smvVol",
    moneyness_axis: bool = False,
) -> go.Figure:
    """
    Heatmap of IV across strike (or log-moneyness) × DTE.

    Parameters
    ----------
    cs : ChainSlice
    iv_col : str
        Column to use for IV values (default ``smvVol``).
    moneyness_axis : bool
        If True, Y-axis shows ``logMny`` instead of raw strike.
    """
    df = cs.df.copy()
    y_col = "logMny" if moneyness_axis else "strike"
    y_label = "Log Moneyness ln(K/S)" if moneyness_axis else "Strike"

    pivot = df.pivot_table(index=y_col, columns="dte", values=iv_col, aggfunc="median")
    pivot = pivot.sort_index(ascending=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=[str(c) for c in pivot.columns],
            y=[f"{v:.1f}" if moneyness_axis else f"{v:.0f}" for v in pivot.index],
            colorscale="Viridis",
            colorbar=dict(title="IV"),
            hovertemplate=(
                "DTE: %{x}<br>"
                + f"{y_label}: " + "%{y}<br>"
                + "IV: %{z:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"IV Surface — {cs.ticker} ({cs.trade_date})",
        xaxis_title="DTE",
        yaxis_title=y_label,
    )
    return fig


# ======================================================================
# Plot 2 — Spread % Heatmap
# ======================================================================

def plot_spread_heatmap(
    cs: ChainSlice,
    side: Literal["call", "put", "worst"] = "worst",
    threshold: float = 0.50,
    moneyness_axis: bool = False,
) -> go.Figure:
    """
    Heatmap of bid-ask spread % across strike × DTE.

    Parameters
    ----------
    side : str
        ``"call"``, ``"put"``, or ``"worst"`` (max of both).
    threshold : float
        Reference spread% drawn as a contour/annotation.
    moneyness_axis : bool
        Y-axis as log-moneyness instead of strike.
    """
    df = cs.df.copy()
    y_col = "logMny" if moneyness_axis else "strike"
    y_label = "Log Moneyness ln(K/S)" if moneyness_axis else "Strike"

    if side == "call":
        val_col = "callSprPct"
        title_side = "Call"
    elif side == "put":
        val_col = "putSprPct"
        title_side = "Put"
    else:
        df["_worstSprPct"] = np.maximum(df["callSprPct"], df["putSprPct"])
        val_col = "_worstSprPct"
        title_side = "Worst-of Call/Put"

    pivot = df.pivot_table(index=y_col, columns="dte", values=val_col, aggfunc="median")
    pivot = pivot.sort_index(ascending=True)

    # Clamp for colour scale (cap at 200%)
    z = np.clip(pivot.values, 0, 2.0)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=[str(c) for c in pivot.columns],
            y=[f"{v:.1f}" if moneyness_axis else f"{v:.0f}" for v in pivot.index],
            colorscale=[[0, "green"], [0.25, "yellow"], [0.5, "orange"], [1, "red"]],
            zmin=0,
            zmax=2.0,
            colorbar=dict(title="Spread %"),
            hovertemplate=(
                "DTE: %{x}<br>"
                + f"{y_label}: " + "%{y}<br>"
                + "Spread%: %{z:.2%}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Spread% Heatmap ({title_side}) — {cs.ticker} ({cs.trade_date})  |  threshold = {threshold:.0%}",
        xaxis_title="DTE",
        yaxis_title=y_label,
    )
    return fig


# ======================================================================
# Plot 3 — OI / Volume by Strike
# ======================================================================

def plot_oi_volume(
    cs: ChainSlice,
    expiry_date: date,
) -> go.Figure:
    """
    Open Interest and Volume by strike for a single expiry (two subplots).
    """
    sub = cs.filter_expiry(expiry_date)
    if sub.empty:
        raise ValueError(f"No data for expiry {expiry_date}")

    sub = sub.sort_values("strike")
    spot = cs.spot_price()
    dte_val = int((expiry_date - cs.trade_date).days)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Open Interest", "Volume"),
        vertical_spacing=0.08,
    )

    # OI traces
    fig.add_trace(
        go.Scatter(x=sub["strike"], y=sub["callOpenInterest"],
                   mode="lines", name="Call OI", line=dict(color="royalblue")),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=sub["strike"], y=sub["putOpenInterest"],
                   mode="lines", name="Put OI", line=dict(color="crimson")),
        row=1, col=1,
    )

    # Volume traces
    fig.add_trace(
        go.Scatter(x=sub["strike"], y=sub["callVolume"],
                   mode="lines", name="Call Vol", line=dict(color="royalblue", dash="dot")),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=sub["strike"], y=sub["putVolume"],
                   mode="lines", name="Put Vol", line=dict(color="crimson", dash="dot")),
        row=2, col=1,
    )

    # ATM vertical lines
    for row in [1, 2]:
        fig.add_vline(
            x=spot, line_dash="dash", line_color="gray",
            annotation_text="ATM" if row == 1 else None,
            row=row, col=1,
        )

    fig.update_layout(
        title=f"OI & Volume — {cs.ticker} ({cs.trade_date})  |  Expiry {expiry_date} ({dte_val} DTE)",
        xaxis2_title="Strike",
        height=600,
    )
    return fig


# ======================================================================
# Plot 4 — Smile / Skew
# ======================================================================

def plot_smile(
    cs: ChainSlice,
    expiry_date: date,
    iv_col: str = "smvVol",
    x_axis: Literal["logMny", "strike", "delta"] = "logMny",
    show_raw_iv: bool = True,
) -> go.Figure:
    """
    IV smile for a single expiry.

    Parameters
    ----------
    x_axis : str
        ``"logMny"``, ``"strike"``, or ``"delta"``
    show_raw_iv : bool
        Overlay callMidIv / putMidIv scatter points alongside smvVol line.
    """
    sub = cs.filter_expiry(expiry_date).sort_values(x_axis)
    if sub.empty:
        raise ValueError(f"No data for expiry {expiry_date}")

    spot = cs.spot_price()
    dte_val = int((expiry_date - cs.trade_date).days)

    x_labels = {
        "logMny": "Log Moneyness ln(K/S)",
        "strike": "Strike",
        "delta": "Call Delta",
    }

    fig = go.Figure()

    # Primary smooth IV line
    fig.add_trace(
        go.Scatter(
            x=sub[x_axis], y=sub[iv_col],
            mode="lines+markers", name=iv_col,
            line=dict(color="royalblue"),
            marker=dict(size=4),
        )
    )

    # Optional raw call/put mid IV overlay
    if show_raw_iv:
        if "callMidIv" in sub.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub[x_axis], y=sub["callMidIv"],
                    mode="markers", name="callMidIv",
                    marker=dict(color="green", size=4, symbol="cross"),
                )
            )
        if "putMidIv" in sub.columns:
            fig.add_trace(
                go.Scatter(
                    x=sub[x_axis], y=sub["putMidIv"],
                    mode="markers", name="putMidIv",
                    marker=dict(color="red", size=4, symbol="x"),
                )
            )

    # ATM reference line
    if x_axis == "logMny":
        fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="ATM")
    elif x_axis == "strike":
        fig.add_vline(x=spot, line_dash="dash", line_color="gray", annotation_text="ATM")
    elif x_axis == "delta":
        fig.add_vline(x=0.50, line_dash="dash", line_color="gray", annotation_text="Δ=0.50")

    fig.update_layout(
        title=f"IV Smile — {cs.ticker} ({cs.trade_date})  |  Expiry {expiry_date} ({dte_val} DTE)",
        xaxis_title=x_labels.get(x_axis, x_axis),
        yaxis_title="Implied Volatility",
    )
    return fig


# ======================================================================
# Plot 5 — Term Structure
# ======================================================================

def plot_term_structure(
    cs: ChainSlice,
    target_delta: float = 0.50,
    iv_col: str = "smvVol",
) -> go.Figure:
    """
    IV term structure at a fixed call delta (interpolated per expiry).
    """
    rows = []
    for exp in cs.available_expiries():
        sub = cs.filter_expiry(exp).dropna(subset=["delta", iv_col]).sort_values("delta")
        x = sub["delta"].values
        y = sub[iv_col].values
        dte = (exp - cs.trade_date).days
        if len(x) < 2 or target_delta < x.min() or target_delta > x.max():
            continue
        iv_interp = float(np.interp(target_delta, x, y))
        rows.append({"expiry": exp, "dte": dte, "iv": iv_interp})

    if not rows:
        fig = go.Figure()
        fig.update_layout(title="Term Structure — no valid expiry data")
        return fig

    ts = pd.DataFrame(rows).sort_values("dte")

    fig = go.Figure(
        go.Scatter(
            x=ts["dte"], y=ts["iv"],
            mode="lines+markers",
            marker=dict(size=8, color="royalblue"),
            text=[str(r["expiry"]) for _, r in ts.iterrows()],
            hovertemplate="DTE: %{x}<br>IV: %{y:.3f}<br>Expiry: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Term Structure @ Δ={target_delta:.2f} — {cs.ticker} ({cs.trade_date})",
        xaxis_title="DTE",
        yaxis_title="Implied Volatility",
    )
    return fig


# ======================================================================
# Plot 6 — Theta per Capital
# ======================================================================

def plot_theta_per_capital(
    cs: ChainSlice,
    expiry_date: date,
    structure: Literal["csp", "straddle"] = "csp",
) -> go.Figure:
    """
    Theta carry efficiency by moneyness for a single expiry.

    Parameters
    ----------
    structure : str
        ``"csp"`` — cash-secured put capital = strike × 100
        ``"straddle"`` — capital = 2 × strike × 100
    """
    sub = cs.filter_expiry(expiry_date).sort_values("logMny").copy()
    if sub.empty:
        raise ValueError(f"No data for expiry {expiry_date}")

    dte_val = int((expiry_date - cs.trade_date).days)
    spot = cs.spot_price()

    # Theta in $ per contract
    sub["theta_dollar"] = sub["theta"].abs() * 100.0  # theta is negative, use abs

    if structure == "csp":
        sub["capital"] = sub["strike"] * 100.0
        cap_label = "CSP (strike×100)"
    else:
        sub["capital"] = sub["strike"] * 200.0
        cap_label = "Straddle (2×strike×100)"

    sub["theta_per_cap"] = sub["theta_dollar"] / np.maximum(sub["capital"], EPS)

    # Expected move band: ±S × IV_ATM × sqrt(T/365)
    atm_row = _atm_row(sub)
    iv_atm = atm_row.get("smvVol", 0.30)
    t_years = max(dte_val / 365.0, 1e-6)
    em = spot * iv_atm * np.sqrt(t_years)
    em_logmny_lo = np.log(max((spot - em) / spot, 1e-6))
    em_logmny_hi = np.log((spot + em) / spot)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["logMny"], y=sub["theta_per_cap"],
            mode="lines+markers",
            marker=dict(size=4, color="royalblue"),
            name="θ / capital",
            hovertemplate="logMny: %{x:.3f}<br>θ/cap: %{y:.6f}<extra></extra>",
        )
    )

    # Expected move band
    fig.add_vrect(
        x0=em_logmny_lo, x1=em_logmny_hi,
        fillcolor="orange", opacity=0.15,
        annotation_text="±1 EM", annotation_position="top left",
        line_width=0,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="ATM")

    fig.update_layout(
        title=f"Theta / Capital ({cap_label}) — {cs.ticker} ({cs.trade_date})  |  {expiry_date} ({dte_val} DTE)",
        xaxis_title="Log Moneyness ln(K/S)",
        yaxis_title="θ$ / Capital$ per day",
    )
    return fig


# ======================================================================
# Plot 7 — Straddle Friction by DTE
# ======================================================================

def plot_straddle_friction(
    cs: ChainSlice,
    threshold: float = 0.12,
) -> go.Figure:
    """
    ATM straddle round-trip cost (friction) for each available expiry.

    friction = (callSpr + putSpr) / (callMid + putMid) at the ATM strike.
    """
    expiries = cs.available_expiries()
    rows = []
    for exp in expiries:
        friction = compute_atm_friction(cs, exp)
        dte = (exp - cs.trade_date).days
        rows.append({"expiry": exp, "dte": dte, "friction": friction})

    if not rows:
        fig = go.Figure()
        fig.update_layout(title="Straddle Friction — no expiry data")
        return fig

    fdf = pd.DataFrame(rows).dropna(subset=["friction"]).sort_values("dte")

    colours = ["green" if f <= threshold else "red" for f in fdf["friction"]]

    fig = go.Figure(
        go.Bar(
            x=fdf["dte"],
            y=fdf["friction"],
            marker_color=colours,
            text=[f"{f:.1%}" for f in fdf["friction"]],
            textposition="outside",
            hovertemplate="DTE: %{x}<br>Friction: %{y:.2%}<br>Expiry: %{customdata}<extra></extra>",
            customdata=[str(e) for e in fdf["expiry"]],
        )
    )
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="gray",
        annotation_text=f"threshold = {threshold:.0%}",
    )
    fig.update_layout(
        title=f"ATM Straddle Friction — {cs.ticker} ({cs.trade_date})",
        xaxis_title="DTE",
        yaxis_title="Friction (round-trip cost %)",
        yaxis_tickformat=".0%",
    )
    return fig


# ======================================================================
# Plot 8 — Cross-Sectional Tradability Distribution
# ======================================================================

def plot_tradability_distribution(
    friction: pd.Series,
    threshold: float = 0.12,
    signal_friction: Optional[pd.Series] = None,
    trade_date: Optional[date] = None,
) -> go.Figure:
    """
    Histogram / CDF of ATM straddle friction across a universe.

    Parameters
    ----------
    friction : pd.Series
        Indexed by ticker, values = friction ratios.
    threshold : float
        Reference threshold line.
    signal_friction : pd.Series, optional
        Subset (e.g. Q10 picks) to overlay.
    trade_date : date, optional
        For title annotation.
    """
    clean = friction.dropna()
    n_pass = (clean <= threshold).sum()
    n_total = len(clean)
    pct_pass = n_pass / max(n_total, 1) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=clean.values,
            nbinsx=50,
            name=f"Universe ({n_total} tickers)",
            marker_color="royalblue",
            opacity=0.7,
        )
    )

    if signal_friction is not None:
        sig_clean = signal_friction.dropna()
        n_sig_pass = (sig_clean <= threshold).sum()
        fig.add_trace(
            go.Histogram(
                x=sig_clean.values,
                nbinsx=30,
                name=f"Signal ({len(sig_clean)} tickers, {n_sig_pass} pass)",
                marker_color="orange",
                opacity=0.7,
            )
        )

    fig.add_vline(
        x=threshold, line_dash="dash", line_color="red",
        annotation_text=f"threshold = {threshold:.0%}",
    )

    date_str = f" — {trade_date}" if trade_date else ""
    fig.update_layout(
        title=f"Tradability Distribution{date_str}  |  {n_pass}/{n_total} pass ({pct_pass:.0f}%)",
        xaxis_title="ATM Straddle Friction",
        yaxis_title="Count",
        xaxis_tickformat=".0%",
        barmode="overlay",
    )
    return fig


# ======================================================================
# Plot 9 — Tradability Time Series
# ======================================================================

def plot_tradability_timeseries(
    friction_df: pd.DataFrame,
    threshold: float = 0.12,
    vix: Optional[pd.Series] = None,
) -> go.Figure:
    """
    Median friction & pass-rate over time (two-panel).

    Parameters
    ----------
    friction_df : pd.DataFrame
        Required columns: ``date``, ``median_friction``, ``pct_passing``.
        Optional: ``std_friction`` for ±1σ band.
    threshold : float
        Reference line on the friction panel.
    vix : pd.Series, optional
        VIX values indexed by date for overlay on secondary axis.
    """
    df = friction_df.sort_values("date").copy()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Median ATM Straddle Friction", "% Tickers Passing Threshold"),
        vertical_spacing=0.10,
        specs=[[{"secondary_y": True}], [{}]],
    )

    # --- Top panel: median friction ---
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["median_friction"],
            mode="lines", name="Median Friction",
            line=dict(color="royalblue"),
        ),
        row=1, col=1,
    )

    # ±1σ band if available
    if "std_friction" in df.columns:
        upper = df["median_friction"] + df["std_friction"]
        lower = (df["median_friction"] - df["std_friction"]).clip(lower=0)
        fig.add_trace(
            go.Scatter(
                x=pd.concat([df["date"], df["date"][::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(65,105,225,0.15)",
                line=dict(width=0),
                name="±1σ",
                showlegend=True,
            ),
            row=1, col=1,
        )

    fig.add_hline(
        y=threshold, line_dash="dash", line_color="red",
        annotation_text=f"threshold = {threshold:.0%}",
        row=1, col=1,
    )

    # VIX overlay
    if vix is not None:
        fig.add_trace(
            go.Scatter(
                x=vix.index, y=vix.values,
                mode="lines", name="VIX",
                line=dict(color="orange", dash="dot"),
            ),
            row=1, col=1, secondary_y=True,
        )
        fig.update_yaxes(title_text="VIX", secondary_y=True, row=1, col=1)

    # --- Bottom panel: pass-rate ---
    fig.add_trace(
        go.Scatter(
            x=df["date"], y=df["pct_passing"],
            mode="lines", name="% Passing",
            line=dict(color="green"),
        ),
        row=2, col=1,
    )
    fig.add_hline(
        y=0.80, line_dash="dash", line_color="gray",
        annotation_text="80%",
        row=2, col=1,
    )

    fig.update_layout(
        title="Tradability Over Time",
        height=650,
    )
    fig.update_yaxes(title_text="Friction", row=1, col=1)
    fig.update_yaxes(title_text="Pass Rate", tickformat=".0%", row=2, col=1)
    return fig


# ======================================================================
# Plot 10 — Signal-Conditional Tradability
# ======================================================================

def plot_signal_tradability(
    universe_friction: pd.Series,
    signal_friction: pd.Series,
    threshold: float = 0.12,
    trade_date: Optional[date] = None,
) -> go.Figure:
    """
    Compare friction distributions: signal-selected tickers vs full universe.

    Renders side-by-side box plots with pass-rate annotations.
    """
    uni_clean = universe_friction.dropna()
    sig_clean = signal_friction.dropna()

    uni_pass = (uni_clean <= threshold).mean() * 100
    sig_pass = (sig_clean <= threshold).mean() * 100

    fig = go.Figure()

    fig.add_trace(
        go.Box(
            y=uni_clean.values,
            name=f"Universe (n={len(uni_clean)}, {uni_pass:.0f}% pass)",
            marker_color="royalblue",
            boxmean=True,
        )
    )
    fig.add_trace(
        go.Box(
            y=sig_clean.values,
            name=f"Signal (n={len(sig_clean)}, {sig_pass:.0f}% pass)",
            marker_color="orange",
            boxmean=True,
        )
    )

    fig.add_hline(
        y=threshold, line_dash="dash", line_color="red",
        annotation_text=f"threshold = {threshold:.0%}",
    )

    date_str = f" — {trade_date}" if trade_date else ""
    fig.update_layout(
        title=f"Signal vs Universe Friction{date_str}",
        yaxis_title="ATM Straddle Friction",
        yaxis_tickformat=".0%",
    )
    return fig
