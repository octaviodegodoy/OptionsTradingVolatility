# -*- coding: utf-8 -*-
"""
GEX Plot Functions — all matplotlib charting for GEX analytics.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_notional_by_strike(vol_by_strike, spot, underlying, show_plots=False):
    """Stacked bar chart of notional volume by strike."""
    vol_by_strike.plot(kind='bar', stacked=True, figsize=(12, 12),
                       color=['#2563EB', '#EF4444'], alpha=0.7)
    plt.axvline(np.argmin(np.abs(vol_by_strike.index - spot)),
                color='black', linestyle='--')
    plt.title(f"Volume Financeiro por Strike — {underlying}")
    plt.ylabel("Volume (R$)")
    plt.xlabel("Strike")
    plt.tight_layout()
    if show_plots:
        plt.show()


def plot_gex_friday(fri_gex_by_strike, spot, underlying, next_friday_str,
                    fri_dte, fri_flip, fri_call_wall, fri_put_wall,
                    show_plots=False):
    """Bar chart of GEX for options expiring next Friday."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_axisbelow(True)
    fri_s = fri_gex_by_strike['Strike'].to_numpy(dtype=float)
    fri_g = (fri_gex_by_strike['GEX_customer'] / 1e6).to_numpy(dtype=float)

    u_fri = np.unique(fri_s)
    if len(u_fri) >= 3:
        bw = np.median(np.diff(u_fri)) * 0.6
    elif len(u_fri) == 2:
        bw = abs(u_fri[1] - u_fri[0]) * 0.6
    else:
        bw = 0.1

    colors = np.where(fri_g >= 0, "#10B981", "#EF4444")
    ax.bar(fri_s, fri_g, width=bw, color=colors,
           edgecolor="none", alpha=0.6, zorder=3)

    if len(fri_g) > 2:
        sm = pd.Series(fri_g).rolling(3, center=True, min_periods=1).mean().values
        ax.plot(fri_s, sm, color='#3B82F6', lw=2, zorder=4, label='Smoothed GEX')

    ax.axvline(spot, color='green', lw=1.2, zorder=5, label=f'Spot: {spot:.2f}')
    if np.isfinite(fri_flip):
        ax.axvline(fri_flip, color='#F59E0B', lw=1.2, ls='--', zorder=5,
                   label=f"Flip: {fri_flip:.2f}")
    if np.isfinite(fri_call_wall):
        ax.axvline(fri_call_wall, color='#2563EB', ls=':', lw=1.6,
                   label=f"Call Wall: {fri_call_wall:.2f}")
        ax.annotate(f"Call Wall\n{fri_call_wall:.2f}",
                    xy=(fri_call_wall, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else 1),
                    xytext=(8, -18), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='#2563EB',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#2563EB', alpha=0.85),
                    ha='left', va='top')
    if np.isfinite(fri_put_wall):
        ax.axvline(fri_put_wall, color='#DC2626', ls='--', lw=1.6,
                   label=f"Put Wall: {fri_put_wall:.2f}")
        ax.annotate(f"Put Wall\n{fri_put_wall:.2f}",
                    xy=(fri_put_wall, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -1),
                    xytext=(-8, 18), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='#DC2626',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#DC2626', alpha=0.85),
                    ha='right', va='bottom')

    cw_str = f"{fri_call_wall:.2f}" if np.isfinite(fri_call_wall) else "N/A"
    pw_str = f"{fri_put_wall:.2f}" if np.isfinite(fri_put_wall) else "N/A"
    ax.set_title(f"{underlying} — GEX Next Friday ({next_friday_str}, {fri_dte} DTE)"
                 f"  |  Call Wall: {cw_str}  |  Put Wall: {pw_str}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('GEX (millions)')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if show_plots:
        plt.show()


def plot_gex_all_expiry(gex_by_strike, spot, underlying, gamma_flip,
                        call_wall, put_wall, show_plots=False):
    """Bar chart of total GEX across all expirations with walls and gamma flip."""
    strikes = gex_by_strike['Strike'].to_numpy(dtype=float)
    gvals = (gex_by_strike['GEX_customer'] / 1e6).to_numpy(dtype=float)

    u = np.unique(strikes)
    if len(u) >= 3:
        step = np.median(np.diff(u))
    elif len(u) == 2:
        step = abs(u[1] - u[0])
    else:
        step = 0.1
    bar_width = step * 0.6

    smooth = pd.Series(gvals).rolling(3, center=True, min_periods=1).mean().values

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_axisbelow(True)

    bar_colors = np.where(gvals >= 0, "#10B981", "#EF4444")
    ax.bar(strikes, gvals, width=bar_width, align="center",
           color=bar_colors, edgecolor="none", alpha=0.55, zorder=3,
           label="Gamma Exposure by Strike")

    ax.plot(strikes, smooth, color="#2563EB", lw=2.2, zorder=4,
            label="Aggregate Gamma Exposure (smoothed)")

    ax.axvline(spot, color="green", lw=1.2, zorder=5, label="Spot")
    if np.isfinite(gamma_flip):
        ax.axvline(gamma_flip, color="#DC2626", lw=1.2, zorder=5,
                   label=f"Gamma Flip (approx): {gamma_flip:.2f}")

    if len(strikes):
        x_min, x_max = strikes.min(), strikes.max()
        if np.isfinite(gamma_flip):
            ax.axvspan(x_min, gamma_flip, color="#E5F3FF", alpha=0.35,
                       label="Positive Gamma: dealers dampen moves")
            ax.axvspan(gamma_flip, x_max, color="#FEE2E2", alpha=0.35,
                       label="Negative Gamma: dealers amplify moves")

    ymin = float(np.nanmin(gvals)) if len(gvals) else -1.0
    ymax = float(np.nanmax(gvals)) if len(gvals) else  1.0
    if ymin < 0 and ymax > 0:
        lim = max(abs(ymin), abs(ymax)) * 1.25
        ax.set_ylim(-lim, lim)
    else:
        pad = 0.15 * (ymax - ymin if ymax > ymin else max(1.0, abs(ymax)))
        ax.set_ylim(ymin - pad, ymax + pad)

    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.2f}"))
    ax.set_xlabel("Strike Price")
    ax.set_ylabel("Gamma Exposure (USD, millions)")
    ax.set_title(f"Gamma Exposure by Strike — {underlying}")
    ax.set_xlim(spot * 0.95, spot * 1.15)
    if np.isfinite(call_wall):
        ax.axvline(call_wall, color="#374151", linestyle=":", lw=1.6,
                   label=f"Call Wall: {call_wall:.2f}")
    if np.isfinite(put_wall):
        ax.axvline(put_wall, color="#9CA3AF", linestyle="--", lw=1.6,
                   label=f"Put Wall: {put_wall:.2f}")

    ax.legend(loc="upper right", ncol=1, fontsize=9, framealpha=0.95)
    fig.text(0.5, 0.96, "om-qs.com", ha="center", va="center", fontsize=9, alpha=0.7)
    ax.grid(alpha=0.25)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    if show_plots:
        plt.show()
