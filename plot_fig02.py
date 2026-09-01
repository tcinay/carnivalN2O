# Plotting Script for Carnival N2O 
# Timur Cinay, 2026
# The Script is designed to be called using plot.sh on with an HPC environment.

# Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd


PLOT_X = np.arange(12, 37) 

# Load Observations 
def load_observations(folder_path):
    dfObs2024 = pd.read_excel(folder_path + 'carnival2024.xlsx')
    dfObs2025 = pd.read_excel(folder_path + 'carnival2025.xlsx')
    dfObs2026 = pd.read_excel(folder_path + 'carnival2026.xlsx')
    return dfObs2024, dfObs2025, dfObs2026

# Load Observations
def load_nonCarnival_observations(folder_path):
    dsObsNonCarnival2024 = xr.open_dataset(folder_path + 'GAL202403_N2O_CO.nc')
    dsObsNonCarnival2025 = xr.open_dataset(folder_path + 'GAL202502_N2O_CO.nc')
    dsObsNonCarnival2026 = xr.open_dataset(folder_path + 'GAL202603_N2O_CO.nc')
    return dsObsNonCarnival2024, dsObsNonCarnival2025, dsObsNonCarnival2026

def clean_column_to_float(col):
    s = col.astype(str)
    s = s.where(~s.str.contains('x'))
    s = s.str.replace('F', '', regex=False)
    s = pd.to_numeric(s, errors='coerce')
    return s

def clean_observations(df):
    df["N2O_C"] = clean_column_to_float(df["N2O_C"])
    df['datetime'] = pd.to_datetime(
        df['date'].astype(str).str.zfill(6) +
        df['time'].astype(str).str.zfill(4),
        format='%y%m%d%H%M'
    ) - pd.Timedelta(hours=6)
    df_clean = df[df['sample'] == '27m_SF_1_air'][['datetime', 'N2O_C']]
    return df_clean


def nc_to_frame(ds, var='n2o_mean', time_name='time', to_local=True):
    df = pd.DataFrame({
        'datetime': pd.to_datetime(ds[time_name].values),
        'N2O_C': np.asarray(ds[var].values, dtype=float).squeeze(),
    })
    if to_local:
        df['datetime'] = df['datetime'] - pd.Timedelta(hours=6)
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


def split_carnival_days(wC):
    day1 = pd.Timestamp(wC[0]).normalize()
    day2 = day1 + pd.Timedelta(days=1)
    return day1, day2


def prepost_days(wC):
    day1, day2 = split_carnival_days(wC)
    return [day1 - pd.Timedelta(days=1), day2 + pd.Timedelta(days=1)]


def noon_window(day):
    t0 = pd.Timestamp(day).normalize() +  pd.Timedelta(hours=12)
    return t0, t0 + pd.Timedelta(days=1)


def day_stats(df, day, detrend=False):
    t0, t1 = noon_window(day)
    return diurnal_stats(df, t0, t1 - pd.Timedelta(seconds=1), detrend=detrend)


def prepost_stats(df, wC, detrend=False):
    mask = pd.Series(False, index=df.index)

    for day in prepost_days(wC):
        t0, t1 = noon_window(day)
        mask |= (df['datetime'] >= t0) & (df['datetime'] < t1)

    return diurnal_stats(df[mask], detrend=detrend)


def print_diurnal_stats(stats, label, table=True):
    s = stats.dropna(subset=['median'])

    if s.empty:
        print(f"  [{label}] no data in window")
        return
    
    print(f"  [{label}] hours covered: {len(s)}/24, n/hr: {s['n'].min():.0f}-{s['n'].max():.0f}")
    print(f"  [{label}] across-hour mean of hourly means:   {s['mean'].mean():.1f} ppb")
    print(f"  [{label}] across-hour mean of hourly medians: {s['median'].mean():.1f} ppb")
    print(f"  [{label}] median hourly IQR width: {s['iqr'].median():.1f} ppb "
          f"(range {s['iqr'].min():.1f}-{s['iqr'].max():.1f} ppb)")
    print(f"  [{label}] peak hour {s['median'].idxmax():02d}:00 median={s['median'].max():.1f} ppb | "
          f"trough hour {s['median'].idxmin():02d}:00 median={s['median'].min():.1f} ppb | "
          f"amplitude={s['median'].max() - s['median'].min():.1f} ppb")
    
    if table:
        print(f"  {'hr':>3} {'n':>4} {'mean':>8} {'median':>8} {'p25':>8} {'p75':>8} {'IQR':>7}")
        for h in np.r_[np.arange(12, 24), np.arange(0, 12)]:                       # print in noon-to-noon order
            if h not in s.index:
                continue
            r = s.loc[h]
            print(f"  {h:>3d} {r['n']:>4.0f} {r['mean']:>8.1f} {r['median']:>8.1f} "
                  f"{r['p25']:>8.1f} {r['p75']:>8.1f} {r['iqr']:>7.1f}")


def print_contrast(statsC, labC, statsR, labR):
    if statsC['median'].dropna().empty or statsR['median'].dropna().empty:
        print(f"  [contrast] {labC} vs {labR}: insufficient data")
        return
    
    ampC = statsC['median'].max() - statsC['median'].min()
    ampR = statsR['median'].max() - statsR['median'].min()
    ratio = ampC / ampR if ampR > 0 else np.nan
    print(f"  [contrast] {labC} - {labR}: offset "
          f"{statsC['median'].mean() - statsR['median'].mean():.1f} ppb | "
          f"amplitude ratio: {ratio:.1f}")


# Plotting Functions
def to_plot_order(series):
    plot_hours = np.r_[np.arange(12, 24), np.arange(0, 12)]
    v = series.reindex(plot_hours).to_numpy(dtype=float)
    return np.r_[v, v[0]]


def plot_diurnal(ax, stats, color, label, stat='median', band='iqr', fs=12,
                 linestyle='-', marker='o', alpha_band=0.18, hatch=None):
    
    x = np.arange(12, 37) 

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


def plot_diurnal_panels(carnival_windows, noncarn_windows, detrend=False, stat='median', band='iqr',  print_hourly_table=True,):
    years = sorted(carnival_windows.keys())
    nrows = len(years)

    fig, axs = plt.subplots(nrows, 1, figsize=(7.0, 2.8 * nrows), squeeze=False)
    axs = axs[:, 0]
    fs = 12

    c_color, n_color = 'firebrick', 'royalblue'   # carnival day 1 / non-carnival
    c2_color, n2_color = 'darkorange', 'darkcyan'  # carnival day 2 / pre-post carnival

    for row, yr in enumerate(years):
        ax = axs[row]
        dfC, wC = carnival_windows[yr]
        dfN, wN = noncarn_windows[yr]
        day1, day2 = split_carnival_days(wC)

        statsC1 = day_stats(dfC, day1, detrend=detrend)
        statsC2 = day_stats(dfC, day2, detrend=detrend)
        statsP  = prepost_stats(dfC, wC, detrend=detrend)

        statsN  = diurnal_stats(dfN, pd.Timestamp(wN[0]).normalize() + pd.Timedelta(hours=12),
                                pd.Timestamp(wN[1]).normalize() + pd.Timedelta(hours=12), detrend=detrend)
        series = [
            (statsN,  n_color,  'Non-carnival',     '--', 's', band, 0.15),
            (statsP,  n2_color,  'Pre/post-carnival (2 d)',  ':',  '^', band, 0.15),
            (statsC1, c_color,  'Carnival (night 1)', '-',  'o', None, 0.0),
            (statsC2, c2_color, 'Carnival (night 2)', '-.', 'D', None, 0.0),
        ]

        for s, c, lab, ls, mk, bd, ab in series:
            plot_diurnal(ax, s, c, lab, stat=stat, band=bd, fs=fs, linestyle=ls, marker=mk, alpha_band=ab)
            
        ax.set_xlim(12, 36)
        ax.set_xticks([12, 18, 24, 30, 36])
        ax.set_xticklabels(['12', '18', '00', '06', '12'])

        for hx in (18, 30):
            ax.axvline(hx, color='0.55', linestyle=':', linewidth=0.9, zorder=0)

        ax.axvline(24, color='0.35', linestyle='-', linewidth=0.8, zorder=0)
        ax.text(0.05, 0.87, yr, transform=ax.transAxes, fontsize=fs, fontweight='bold')
        ax.tick_params(axis='both', labelsize=fs)
        ax.set_ylabel(r'$\mathrm{N_2O}$ (ppb)', fontsize=fs)

        if row == nrows - 1:
            ax.set_xlabel('Hour of day (GMT-6)', fontsize=fs)
        if row == 0:
            ax.legend(fontsize=fs-3, loc='upper right', framealpha=0.9, ncol=2)

        # Stats
        print(f"--- diurnal {yr} (noon-to-noon) ---")
        print_diurnal_stats(statsC1, f'carnival night 1 ({day1:%m-%d} 12:00 -> {day2:%m-%d} 12:00)', table=print_hourly_table)
        
        day = pd.Timedelta(days=1)
        print_diurnal_stats(statsC2, f'carnival night 2 ({day2:%m-%d} 12:00 -> ' f'{day2 + day:%m-%d} 12:00)', table=print_hourly_table)

        pp = prepost_days(wC)
        print_diurnal_stats(statsP, f'pre/post ({pp[0]:%m-%d} 12:00, {pp[1]:%m-%d} 12:00)', table=print_hourly_table)

        print_diurnal_stats(statsN, 'non-carnival', table=print_hourly_table)
        print_contrast(statsC1, 'night 1', statsN, 'non-carnival')
        print_contrast(statsC2, 'night 2', statsN, 'non-carnival')
        print_contrast(statsC1, 'night 1', statsP, 'pre/post')
        print_contrast(statsC2, 'night 2', statsP, 'pre/post')

    for k, lab in enumerate(['(a)', '(b)', '(c)', '(d)'][:nrows]):
        axs[k].text(-0.18, 1.00, lab, transform=axs[k].transAxes, fontsize=fs, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname('figures/fig02_diurnal.pdf'), exist_ok=True)
    plt.savefig('figures/fig02_diurnal.pdf', dpi=300)


# Main Execution
if __name__ == "__main__":
    obs_path     = './inputs/'
    noncarn_path = '../../inversions/data/obs_data/galapagos/'

    # Carnival observations
    dfObs2024, dfObs2025, dfObs2026 = load_observations(obs_path)
    dfObs2024 = clean_observations(dfObs2024)
    dfObs2025 = clean_observations(dfObs2025)
    dfObs2026 = clean_observations(dfObs2026)

    # Non-carnival observations
    dsNC2024, dsNC2025, dsNC2026 = load_nonCarnival_observations(noncarn_path)
    dfNonCarn2024 = nc_to_frame(dsNC2024)
    dfNonCarn2025 = nc_to_frame(dsNC2025)
    dfNonCarn2026 = nc_to_frame(dsNC2026)

    # Carnival Dates
    dates2024 = pd.to_datetime(['2024-02-12', '2024-02-14'])
    dates2025 = pd.to_datetime(['2025-03-03', '2025-03-05'])
    dates2026 = pd.to_datetime(['2026-02-16', '2026-02-18'])

    # Non-Carnival Null Hypothesis Dates
    datesNonCarnival2024 = pd.to_datetime(['2024-03-01', '2024-03-07'])
    datesNonCarnival2025 = pd.to_datetime(['2025-02-10', '2025-02-16'])
    datesNonCarnival2026 = pd.to_datetime(['2026-03-01', '2026-03-07'])
    carnival_windows = {
        '2024': (dfObs2024, dates2024),
        '2025': (dfObs2025, dates2025),
        '2026': (dfObs2026, dates2026),
    }

    noncarn_windows = {
        '2024': (dfNonCarn2024, datesNonCarnival2024),
        '2025': (dfNonCarn2025, datesNonCarnival2025),
        '2026': (dfNonCarn2026, datesNonCarnival2026),
    }

    plot_diurnal_panels(carnival_windows, noncarn_windows, detrend=False, stat='median', band='iqr', print_hourly_table=True,)