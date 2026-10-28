# Plotting Script for Carnival N2O
# Timur Cinay, 2026
# The Script is designed to be called using plot.sh on with an HPC environment.

# Import Libraries
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Define Run Configurations
# Input Paths & File Formatting
config_obs_path = './inputs/'
config_fig_path = './figures/'
config_years = ['2024', '2025', '2026']

# Non-Carnival Observations (BCO-DMO, https://www.bco-dmo.org/dataset/917743)
config_noncarn_files = {'2024': 'GAL202403_N2O_CO.nc', '2025': 'GAL202502_N2O_CO.nc', '2026': 'GAL202603_N2O_CO.nc'}

# Unit & Time Conversions
config_utc_offset = pd.Timedelta(hours=6)  # Galapagos is GMT-6

# Carnival Dates (Local, GMT-6; the first entry anchors carnival night 1)
config_carnival_dates = {
    '2024': pd.to_datetime(['2024-02-12', '2024-02-14']),
    '2025': pd.to_datetime(['2025-03-03', '2025-03-05']),
    '2026': pd.to_datetime(['2026-02-16', '2026-02-18']),
}

# Non-Carnival Null Hypothesis Dates (Local, GMT-6)
config_noncarn_dates = {
    '2024': pd.to_datetime(['2024-03-01', '2024-03-07']),
    '2025': pd.to_datetime(['2025-02-10', '2025-02-16']),
    '2026': pd.to_datetime(['2026-03-01', '2026-03-07']),
}

# Noon-to-Noon Plotting Convention
# A "day" runs 12:00 -> 12:00 (next day), so midnight sits at the centre.
# Hour-of-day bins stay 0-23; only the plotting order / x-coordinate changes.
noon = pd.Timedelta(hours=12)
one_day = pd.Timedelta(days=1)
plot_hours = np.r_[np.arange(12, 24), np.arange(0, 12)]   # 12..23, 00..11
plot_x = np.arange(12, 37)                                 # 12..36 (36 == 12 wrap)

# Load Observations (Data S1, one sheet per year, UTC)
def load_observations(folder_path, filename='DataS1.xlsx', to_local=True):
    sheets = pd.read_excel(folder_path + filename, sheet_name=config_years)
    out = []
    for year in config_years:
        df = sheets[year][['datetime', 'N2O_C']].copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        if to_local:
            df['datetime'] = df['datetime'] - config_utc_offset
        out.append(df.sort_values('datetime').reset_index(drop=True))
    return out[0], out[1], out[2]

# Load Non-Carnival Observations (netCDF, UTC)
def load_nonCarnival_observations(folder_path):
    out = []
    for year in config_years:
        with xr.open_dataset(folder_path + config_noncarn_files[year]) as ds:
            out.append(nc_to_frame(ds))
    return out[0], out[1], out[2]

def nc_to_frame(ds, var='n2o_mean', time_name='time', to_local=True):
    df = pd.DataFrame({
        'datetime': pd.to_datetime(ds[time_name].values),
        'N2O_C': np.asarray(ds[var].values, dtype=float).squeeze(),
    })
    if to_local:
        df['datetime'] = df['datetime'] - config_utc_offset
    return df.dropna(subset=['N2O_C'])

# Diurnal Statistics
def diurnal_stats(df, t0=None, t1=None, value='N2O_C', detrend=False):
    d = df.dropna(subset=[value]).copy()
    if t0 is not None:
        d = d[d['datetime'] >= t0]
    if t1 is not None:
        d = d[d['datetime'] <= t1]
    if detrend:
        d[value] = d[value] - d[value].median()

    d['hour'] = d['datetime'].dt.hour
    g = d.groupby('hour')[value]

    out = pd.DataFrame({
        'mean':   g.mean(),
        'median': g.median(),
        'std':    g.std(),
        'p25':    g.quantile(0.25),
        'p75':    g.quantile(0.75),
        'n':      g.size(),
    }).reindex(range(24))

    out['iqr'] = out['p75'] - out['p25']

    return out

# Carnival Day / Pre-Post Day Windows Derived from the Carnival Window
def split_carnival_days(wC):
    day1 = pd.Timestamp(wC[0]).normalize()
    day2 = day1 + one_day
    return day1, day2

def prepost_days(wC):
    day1, day2 = split_carnival_days(wC)
    return [day1 - one_day, day2 + one_day]

def noon_window(day):
    t0 = pd.Timestamp(day).normalize() + noon
    return t0, t0 + one_day

def day_stats(df, day, detrend=False):
    t0, t1 = noon_window(day)
    return diurnal_stats(df, t0, t1 - pd.Timedelta(seconds=1), detrend=detrend)

def prepost_stats(df, wC, detrend=False):
    mask = pd.Series(False, index=df.index)

    for day in prepost_days(wC):
        t0, t1 = noon_window(day)
        mask |= (df['datetime'] >= t0) & (df['datetime'] < t1)

    return diurnal_stats(df[mask], detrend=detrend)

# Plotting Functions
# Reindex an Hour-of-Day (0-23) Series to 12 -> 35 Order and Close the Curve
def to_plot_order(series):
    v = series.reindex(plot_hours).to_numpy(dtype=float)
    return np.r_[v, v[0]]

def plot_diurnal(ax, stats, color, label, stat='median', band='iqr', fs=12,
                 linestyle='-', marker='o', alpha_band=0.18, hatch=None):
    x = plot_x
    center = to_plot_order(stats[stat])

    if band == 'iqr':
        lo = to_plot_order(stats['p25'])
        hi = to_plot_order(stats['p75'])
    elif band == 'std':
        lo = to_plot_order(stats[stat] - stats['std'])
        hi = to_plot_order(stats[stat] + stats['std'])
    else:
        lo = hi = None
    ax.plot(x, center, color=color, linestyle=linestyle, marker=marker, ms=3.5, lw=1.6, label=label)
    if lo is not None:
        ax.fill_between(x, lo, hi, facecolor=color, edgecolor=color, alpha=alpha_band,
                        linewidth=0.0 if hatch is None else 0.6, hatch=hatch)

def plot_diurnal_panels(carnival_windows, noncarn_windows, detrend=False, stat='median', band='iqr'):
    years = sorted(carnival_windows.keys())
    nrows = len(years)

    fig, axs = plt.subplots(nrows, 1, figsize=(7.0, 2.8 * nrows), squeeze=False)
    axs = axs[:, 0]
    fs = 12

    c_color, n_color = 'firebrick', 'royalblue'    # carnival night 1 / non-carnival
    c2_color, p_color = 'darkorange', 'darkcyan'   # carnival night 2 / pre-post carnival

    for row, yr in enumerate(years):
        ax = axs[row]
        dfC, wC = carnival_windows[yr]
        dfN, wN = noncarn_windows[yr]
        day1, day2 = split_carnival_days(wC)

        statsC1 = day_stats(dfC, day1, detrend=detrend)
        statsC2 = day_stats(dfC, day2, detrend=detrend)
        statsP  = prepost_stats(dfC, wC, detrend=detrend)

        # Non-Carnival Window Shifted to Noon-to-Noon as Well
        statsN  = diurnal_stats(dfN, pd.Timestamp(wN[0]).normalize() + noon,
                                pd.Timestamp(wN[1]).normalize() + noon, detrend=detrend)
        series = [
            (statsN,  n_color,  'Non-carnival',            '--', 's', band, 0.15),
            (statsP,  p_color,  'Pre/post-carnival (2 d)', ':',  '^', band, 0.15),
            (statsC1, c_color,  'Carnival (night 1)',      '-',  'o', None, 0.0),
            (statsC2, c2_color, 'Carnival (night 2)',      '-.', 'D', None, 0.0),
        ]

        for s, c, lab, ls, mk, bd, ab in series:
            plot_diurnal(ax, s, c, lab, stat=stat, band=bd, fs=fs, linestyle=ls, marker=mk, alpha_band=ab)

        ax.set_xlim(12, 36)
        ax.set_xticks([12, 18, 24, 30, 36])
        ax.set_xticklabels(['12', '18', '00', '06', '12'])

        for hx in (18, 30):  # dusk / dawn markers
            ax.axvline(hx, color='0.55', linestyle=':', linewidth=0.9, zorder=0)

        ax.axvline(24, color='0.35', linestyle='-', linewidth=0.8, zorder=0)  # midnight
        ax.text(0.05, 0.87, yr, transform=ax.transAxes, fontsize=fs, fontweight='bold')
        ax.tick_params(axis='both', labelsize=fs)
        ax.set_ylabel(r'$\mathrm{N_2O}$ (ppb)', fontsize=fs)

        if row == nrows - 1:
            ax.set_xlabel('Hour of day (GMT-6)', fontsize=fs)
        if row == 0:
            ax.legend(fontsize=fs-3, loc='upper right', framealpha=0.9, ncol=2)

    # Add Subplot Labels
    for k, lab in enumerate(['(a)', '(b)', '(c)', '(d)'][:nrows]):
        axs[k].text(-0.18, 1.00, lab, transform=axs[k].transAxes, fontsize=fs, fontweight='bold')

    plt.tight_layout()
    os.makedirs(config_fig_path, exist_ok=True)
    plt.savefig(f'{config_fig_path}fig02_diurnal.pdf', dpi=300)
    plt.close(fig)

# Main Execution
if __name__ == "__main__":
    # Carnival Observations
    dfObs2024, dfObs2025, dfObs2026 = load_observations(config_obs_path)

    # Non-Carnival Observations
    dfNonCarn2024, dfNonCarn2025, dfNonCarn2026 = load_nonCarnival_observations(config_obs_path)

    carnival_windows = {
        '2024': (dfObs2024, config_carnival_dates['2024']),
        '2025': (dfObs2025, config_carnival_dates['2025']),
        '2026': (dfObs2026, config_carnival_dates['2026']),
    }

    noncarn_windows = {
        '2024': (dfNonCarn2024, config_noncarn_dates['2024']),
        '2025': (dfNonCarn2025, config_noncarn_dates['2025']),
        '2026': (dfNonCarn2026, config_noncarn_dates['2026']),
    }

    plot_diurnal_panels(carnival_windows, noncarn_windows, detrend=False, stat='median', band='iqr')
