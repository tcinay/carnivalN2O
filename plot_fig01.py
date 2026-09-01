# Plotting Script for Carnival N2O 
# Timur Cinay, 2026
# The Script is designed to be called using plot.sh on with an HPC environment.

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import xarray as xr
import pandas as pd

# Load Observations 
def load_observations(folder_path, filename='carnival_obs.xlsx', to_local=True):
    sheets = pd.read_excel(folder_path + filename, sheet_name=['2024', '2025', '2026'])
    out = []
    for year in ['2024', '2025', '2026']:
        df = sheets[year].copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        if to_local:
                df['datetime'] = df['datetime'] - pd.Timedelta(hours=6)

        out.append(df.sort_values('datetime').reset_index(drop=True))
    return out[0], out[1], out[2]

def nc_to_frame(ds, var='n2o_mean', time_name='time', to_local=True):
    df = pd.DataFrame({
        'datetime': pd.to_datetime(ds[time_name].values),
        'N2O_C': np.asarray(ds[var].values, dtype=float).squeeze(),
    })
    if to_local:
        df['datetime'] = df['datetime'] - pd.Timedelta(hours=6)
    return df.dropna(subset=['N2O_C'])

# Load STILT-Inversion Results
def load_inversion_results(folder_path):
    ds2024 = xr.open_dataset(folder_path + 'inversionResults_2024.nc')
    ds2025 = xr.open_dataset(folder_path + 'inversionResults_2025.nc')

    ds2024['time'] = ds2024['time'] - pd.Timedelta('6h')
    ds2025['time'] = ds2025['time'] - pd.Timedelta('6h')

    return ds2024, ds2025


# Print Statistics
def print_stats(df, year):
    dates2024 = pd.to_datetime(['2024-02-10', '2024-02-16'])
    dates2025 = pd.to_datetime(['2025-03-01', '2025-03-07'])
    dates2026 = pd.to_datetime(['2026-02-14', '2026-02-20'])

    carn2024  = pd.to_datetime(['2024-02-12', '2024-02-14'])
    carn2025  = pd.to_datetime(['2025-03-03', '2025-03-05'])
    carn2026  = pd.to_datetime(['2026-02-16', '2026-02-18'])

    if year == '2024':
        df = df[(df['datetime'] >= dates2024[0]) & (df['datetime'] <= dates2024[1])]
        carn = carn2024
    elif year == '2025':
        df = df[(df['datetime'] >= dates2025[0]) & (df['datetime'] <= dates2025[1])]
        carn = carn2025
    elif year == '2026':
        df = df[(df['datetime'] >= dates2026[0]) & (df['datetime'] <= dates2026[1])]
        carn = carn2026
    x = df['N2O_C'].dropna()
    bg = df[(df['datetime'] < carn[0]) | (df['datetime'] > carn[1])]['N2O_C'].dropna()
    peak = x.max()
    max_idx = x.idxmax()
    print(f"--- {year} ---")
    print_series_stats(x,  'incl. carnival')
    print_series_stats(bg, 'excl. carnival')
    print(f"Max N2O: {peak:.1f} ppb at {df.loc[max_idx, 'datetime']}")
    print_zscore(peak, x,  'incl. carnival')
    print_zscore(peak, bg, 'excl. carnival')

def print_series_stats(x, label):
    if len(x) == 0:
        print(f"[{label}] no valid samples")
        return
    p25, p75 = x.quantile(0.25), x.quantile(0.75)
    print(f"[{label}] Observations: {len(x)} samples, mean={x.mean():.1f} ppb, std={x.std():.1f} ppb")
    print(f"[{label}] Median N2O: {x.median():.1f} ppb")
    print(f"[{label}] IQR N2O: {p25:.1f}-{p75:.1f} ppb (width {p75 - p25:.1f} ppb)")
    print(f"[{label}] Range N2O: {x.min():.1f}-{x.max():.1f} ppb")

def print_zscore(peak, ref, label):
    if len(ref) < 3:
        print(f"[{label}] Baseline: {len(ref)} samples, too few for z-score")
        return
    ref_mean, ref_std = ref.mean(), ref.std(ddof=1)
    ref_med = ref.median()
    mad = (ref - ref_med).abs().median()
    z = (peak - ref_mean) / ref_std if ref_std > 0 else np.nan
    z_robust = (peak - ref_med) / (1.4826 * mad) if mad > 0 else np.nan
    print(f"[{label}] Peak enhancement over baseline mean: {peak - ref_mean:.1f} ppb")
    print(f"[{label}] Z-score (mean/std): {z:.1f}")
    print(f"[{label}] Robust z-score (median/MAD): {z_robust:.1f}")

# Plotting Function
def plot_mole_fractions(dfObs2024, dfObs2025, dfObs2026, ds2024, ds2025):
    fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=False)
    fs = 12

    # Carnival Dates
    dates2024 = pd.to_datetime(['2024-02-12', '2024-02-14'])
    dates2025 = pd.to_datetime(['2025-03-03', '2025-03-05'])
    dates2026 = pd.to_datetime(['2026-02-16', '2026-02-18'])
    
    # Filter datasets to date range
    ds2024 = ds2024.where((ds2024['time'] >= dates2024[0]) & (ds2024['time'] <= dates2024[1]), drop=True)
    ds2025 = ds2025.where((ds2025['time'] >= dates2025[0]) & (ds2025['time'] <= dates2025[1]), drop=True)

    # Plot Carnival Timeseries
    # 2024
    ax = axs[0]
    ax.plot(dfObs2024['datetime'], dfObs2024['N2O_C'], linestyle='-', color='black', label='Observations')
    ax.plot(ds2024['time'], ds2024['ypred_mean'], color='royalblue', linestyle='-', label='Model-Predicted')
    ax.axvspan(dates2024[0], dates2024[1], color='lightblue', alpha=0.3,)
    ax.text(dates2024[0] - pd.Timedelta(days=1.7), 395, '2024', color='black', fontsize=fs, fontweight='bold')
    ax.set_xlim(dates2024[0] - pd.Timedelta(days=2), dates2024[1] + pd.Timedelta(days=2))
    ax.tick_params(axis='both', labelsize=fs)

    # 2025
    ax = axs[1]
    ax.plot(dfObs2025['datetime'], dfObs2025['N2O_C'], linestyle='-', color='black', label='Observations')
    ax.plot(ds2025['time'], ds2025['ypred_mean'], color='royalblue', linestyle='-', label='Model-Predicted')
    ax.axvspan(dates2025[0], dates2025[1], color='lightblue', alpha=0.3)
    ax.text(dates2025[0] - pd.Timedelta(days=1.7), 395, '2025', color='black', fontsize=fs, fontweight='bold')
    ax.set_xlim(dates2025[0] - pd.Timedelta(days=2), dates2025[1] + pd.Timedelta(days=2))
    ax.tick_params(axis='both', labelsize=fs)

    # 2026
    ax = axs[2]
    ax.plot(dfObs2026['datetime'], dfObs2026['N2O_C'], linestyle='-', color='black', label='Observations')
    ax.axvspan(dates2026[0], dates2026[1], color='lightblue', alpha=0.3)
    ax.text(dates2026[0] - pd.Timedelta(days=1.7), 395, '2026', color='black', fontsize=fs, fontweight='bold')
    ax.set_xlim(dates2026[0] - pd.Timedelta(days=2), dates2026[1] + pd.Timedelta(days=2))
    ax.tick_params(axis='both', labelsize=fs)
    #ax.set_ylim(335, 360)

    # Formatting
    for i in range(len(axs)):
        ax = axs[i]
        ax.set_ylabel(r'$\mathrm{N_2O}$ (ppb)', fontsize=fs)
        ax.set_ylim(335, 410)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        if i == len(axs) - 1:
            ax.set_xlabel('Time (GMT-6)', fontsize=fs)
        if i == 0:
            ax.legend(fontsize=fs-2, loc='upper right')
    # Add subplot labels
    for k, lab in enumerate(['(a)', '(b)', '(c)']):
        axs[k].text(-0.07, 1.00, lab, transform=axs[k].transAxes, fontsize=fs, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig01_molefractions.pdf', dpi=300)

# Main Execution
if __name__ == "__main__":
    folder_path = './inputs/'
    dfObs2024, dfObs2025, dfObs2026 = load_observations(folder_path)

    folder_path = './outputs/'
    ds2024, ds2025 = load_inversion_results(folder_path)

    print_stats(dfObs2024, '2024')
    print_stats(dfObs2025, '2025')
    print_stats(dfObs2026, '2026')

    plot_mole_fractions(dfObs2024, dfObs2025, dfObs2026, ds2024, ds2025)