# Plotting Script for Carnival N2O
# Timur Cinay, 2026
# The Script is designed to be called using plot.sh on with an HPC environment.

# Import Libraries
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define Run Configurations
# Input Paths & File Formatting
config_obs_path = './inputs/'
config_inv_path = './outputs/'
config_fig_path = './figures/'
config_years = ['2024', '2025', '2026']

# Unit & Time Conversions
config_utc_offset = pd.Timedelta(hours=6)  # Galapagos is GMT-6

# Carnival Dates (Local, GMT-6) and the +/-2 d Plotting Window
config_carnival_dates = {
    '2024': pd.to_datetime(['2024-02-12', '2024-02-14']),
    '2025': pd.to_datetime(['2025-03-03', '2025-03-05']),
    '2026': pd.to_datetime(['2026-02-16', '2026-02-18']),
}
config_pad = pd.Timedelta(days=2)

# CO Enhancement Strip Settings
# A bin is enhanced if its mean CO exceeds the +/-2 d window mean + config_co_k_std * std
config_co_bin = '30min'
config_co_k_std = 2.0

# Gaps Longer than This are Left Blank in the Time Series
config_max_gap = pd.Timedelta(minutes=5)

# Load Observations (Data S1, one sheet per year, UTC)
def load_observations(folder_path, filename='DataS1.xlsx', to_local=True):
    sheets = pd.read_excel(folder_path + filename, sheet_name=config_years)
    out = []
    for year in config_years:
        df = sheets[year][['datetime', 'N2O_C', 'CO_C']].copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        if to_local:
            df['datetime'] = df['datetime'] - config_utc_offset
        out.append(df.sort_values('datetime').reset_index(drop=True))
    return out[0], out[1], out[2]

# Load STILT-Inversion Results
def load_inversion_results(folder_path):
    ds2024 = xr.open_dataset(folder_path + 'inversionResults_2024.nc')
    ds2025 = xr.open_dataset(folder_path + 'inversionResults_2025.nc')

    ds2024['time'] = ds2024['time'] - config_utc_offset
    ds2025['time'] = ds2025['time'] - config_utc_offset

    return ds2024, ds2025

# Print Statistics
# Peak N2O and its median/IQR within carnival +/-2 d, and the z-score of the peak
# against the +/-2 d window excluding carnival
def print_stats(df, year):
    c0, c1 = config_carnival_dates[year]
    w0, w1 = c0 - config_pad, c1 + config_pad

    t = df['datetime']
    in_carn = (t >= c0) & (t <= c1)
    in_wide = (t >= w0) & (t <= w1)

    x = df.loc[in_wide, 'N2O_C'].dropna()
    i_peak = x.idxmax()
    peak, t_peak = x.loc[i_peak], df.loc[i_peak, 'datetime']

    ref = df.loc[in_wide & ~in_carn, 'N2O_C'].dropna()
    mean, std = ref.mean(), ref.std(ddof=1)
    z = (peak - mean) / std if std > 0 else np.nan

    print(f"--- {year} ---")
    print(f"Max N2O: {peak:.1f} ppb at {t_peak} (GMT-6)")
    print(f"Median N2O ({w0:%m/%d}-{w1:%m/%d}): {x.median():.1f} ppb (IQR {x.quantile(0.25):.1f}-{x.quantile(0.75):.1f} ppb)")
    print(f"Z-score vs. +/-2 d excl. carnival: {z:.1f}")

# CO Enhancement Flags
# Returns bin edges and flags (1 = enhanced, 0 = not enhanced, NaN = no CO data)
def co_enhancement_flags(df, year):
    c0, c1 = config_carnival_dates[year]
    w0, w1 = c0 - config_pad, c1 + config_pad
    co = df.set_index('datetime')['CO_C'].dropna()
    co = co[(co.index >= w0) & (co.index < w1)]

    # Threshold from the full +/-2 d window, carnival included
    baseline, spread = co.mean(), co.std(ddof=1)
    thresh = baseline + config_co_k_std * spread

    # Bin-Mean CO
    edges = pd.date_range(w0, w1, freq=config_co_bin)
    cats = pd.cut(co.index, bins=edges, right=False)
    co_bin = co.groupby(cats, observed=False).mean().to_numpy()
    flags = np.where(np.isnan(co_bin), np.nan, (co_bin > thresh).astype(float))

    return edges, flags

# Plotting Helpers
# Insert NaN Rows at Data Gaps so Lines Are Not Drawn Across Them
def break_gaps(df, max_gap=config_max_gap):
    gap = df['datetime'].diff() > max_gap
    fill = df.loc[gap, ['datetime']].copy()
    fill['datetime'] = fill['datetime'] - pd.Timedelta(seconds=1)
    return pd.concat([df, fill]).sort_values('datetime').reset_index(drop=True)

# Thin Strip Above the Panel: Black = CO Enhanced, White = Not Enhanced, Gray = No Data
def add_co_strip(ax, edges, flags, fs):
    divider = make_axes_locatable(ax)
    ax_strip = divider.append_axes('top', size='7%', pad=0.04, sharex=ax)
    cmap = ListedColormap(['white', 'black'])
    cmap.set_bad('0.8')
    ax_strip.pcolormesh(mdates.date2num(edges), [0, 1], np.ma.masked_invalid(flags[None, :]),
                        cmap=cmap, vmin=0, vmax=1, shading='flat')
    ax_strip.set_yticks([])
    ax_strip.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax_strip.text(-0.01, 0.5, 'CO', transform=ax_strip.transAxes, ha='right', va='center', fontsize=fs-3)
    return ax_strip

# Vertical Dashed Lines at 16:00 and 00:00 Local on the First Carnival Evening
def mark_first_evening(ax, carnival_start, hour_start=16, hour_end=24):
    day0 = carnival_start.normalize()
    for t in (day0 + pd.Timedelta(hours=hour_start), day0 + pd.Timedelta(hours=hour_end)):
        ax.axvline(t, color='dimgray', linestyle='--', linewidth=1.0, zorder=1)

# Plotting Function
def plot_mole_fractions(dfObs_all, ds_all):
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=False)
    fs = 12
    ylim = (335, 410)

    strips = []
    for ax, year, dfObs in zip(axs, config_years, dfObs_all):
        c0, c1 = config_carnival_dates[year]
        dfPlot = break_gaps(dfObs)

        ax.plot(dfPlot['datetime'], dfPlot['N2O_C'], linestyle='-', color='black', label='Observations')
        if year in ds_all:
            ypred = ds_all[year]['ypred_mean'].sel(time=slice(c0, c1))
            ax.plot(ypred['time'], ypred, color='royalblue', linestyle='-', label='Inversion-Based Estimate')
        ax.axvspan(c0, c1, color='lightblue', alpha=0.3)
        mark_first_evening(ax, c0)
        ax.text(c0 - pd.Timedelta(days=1.7), 395, year, color='black', fontsize=fs, fontweight='bold')
        ax.set_xlim(c0 - config_pad, c1 + config_pad)
        ax.tick_params(axis='both', labelsize=fs)

        # CO Enhancement Strip Above Each Panel
        edges, flags = co_enhancement_flags(dfObs, year)
        strips.append(add_co_strip(ax, edges, flags, fs))

    # Formatting
    for i, ax in enumerate(axs):
        ax.set_ylabel(r'$\mathrm{N_2O}$ (ppb)', fontsize=fs)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        if i == len(axs) - 1:
            ax.set_xlabel('Time (GMT-6)', fontsize=fs)
        if i == 0:
            ax.legend(fontsize=fs-2, loc='upper right')

    # Add Subplot Labels (on the strips, at the top of each panel group)
    for k, lab in enumerate(['(a)', '(b)', '(c)']):
        strips[k].text(-0.07, 1.00, lab, transform=strips[k].transAxes, fontsize=fs, fontweight='bold', va='bottom')

    plt.tight_layout()
    os.makedirs(config_fig_path, exist_ok=True)
    plt.savefig(f'{config_fig_path}fig01_molefractions.pdf', dpi=300)
    plt.close(fig)

# Main Execution
if __name__ == "__main__":
    dfObs2024, dfObs2025, dfObs2026 = load_observations(config_obs_path)
    ds2024, ds2025 = load_inversion_results(config_inv_path)

    print_stats(dfObs2024, '2024')
    print_stats(dfObs2025, '2025')
    print_stats(dfObs2026, '2026')

    plot_mole_fractions([dfObs2024, dfObs2025, dfObs2026], {'2024': ds2024, '2025': ds2025})
