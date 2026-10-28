# Plotting Script for Carnival N2O
# Timur Cinay, 2026
# The Script is designed to be called using plot.sh on with an HPC environment.

# Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define Run Configurations
# Input Paths & File Formatting
config_obs_path = './inputs/'
config_met_path = './inputs/'
config_fig_path = './figures/'
config_years = ['2024', '2025', '2026']

# Meteorological Station Data (available upon request, https://weathergsc.usfq.edu.ec/)
config_met_files = {'2024': '202402.csv', '2025': '202503.csv', '2026': '202602.csv'}

# Unit & Time Conversions
config_utc_offset = pd.Timedelta(hours=6)  # Galapagos is GMT-6

# Carnival Dates (Local, GMT-6)
config_carnival_dates = {
    '2024': pd.to_datetime(['2024-02-12', '2024-02-14']),
    '2025': pd.to_datetime(['2025-03-03', '2025-03-05']),
    '2026': pd.to_datetime(['2026-02-16', '2026-02-18']),
}

# Evening Window (Local Hour, End Exclusive) and Histogram Bin Width (degrees)
config_hour_start = 16
config_hour_end = 24
config_bin_width = 10

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

# Load MET Data (already in GMT-6, no conversion needed)
def load_met(folder_path, filename):
    met = pd.read_csv(folder_path + filename, skiprows=1).iloc[:, 1:]  # drop leading # index col
    met = met.rename(columns={'Date Time GMT-06:00': 'datetime'})
    met['datetime'] = pd.to_datetime(met['datetime'], format='mixed')
    met['Wind Direction'] = pd.to_numeric(met['Wind Direction'], errors='coerce')
    met = met[['datetime', 'Wind Direction']].dropna()
    return met.sort_values('datetime').reset_index(drop=True)

# Select Carnival Evenings (config_hour_start to config_hour_end, Local)
def select_evenings(df, carn_dates):
    t = df['datetime']
    hour = t.dt.hour + t.dt.minute / 60
    in_carnival = (t >= carn_dates[0]) & (t < carn_dates[1])
    in_evening = (hour >= config_hour_start) & (hour < config_hour_end)
    return df[in_carnival & in_evening]

# Match each N2O Observation to the Nearest Raw Wind Direction Reading
def match_wind_direction(dfObs, met):
    dt = met['datetime'].diff().median()  # MET logging interval, used as the match tolerance
    obs = dfObs.dropna(subset=['N2O_C']).sort_values('datetime').copy()
    obs['datetime'] = obs['datetime'].astype('datetime64[ns]')
    met = met.copy()
    met['datetime'] = met['datetime'].astype('datetime64[ns]')
    matched = pd.merge_asof(obs, met, on='datetime', direction='nearest', tolerance=dt)
    return matched.dropna(subset=['Wind Direction']), dt

# Plotting Function
def plot_wind_histograms(dfObs_all, met_all):
    fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=False)
    fs = 12
    n2o_color = 'firebrick'
    bins = np.arange(0, 360 + config_bin_width, config_bin_width)

    # Select Carnival Evenings and Match Wind Direction to Observations
    wd_all, matched_all = [], []
    for year, dfObs, met in zip(config_years, dfObs_all, met_all):
        carn = config_carnival_dates[year]
        wd = select_evenings(met, carn)['Wind Direction']
        matched, _ = match_wind_direction(select_evenings(dfObs, carn), met)
        wd_all.append(wd)
        matched_all.append(matched)

    # Shared Histogram Limit (percent of observations, comparable across years)
    hist_top = max([100 * np.histogram(wd, bins=bins)[0].max() / len(wd) for wd in wd_all if len(wd) > 0]) * 1.15
    n2o_top = max([410] + [matched['N2O_C'].max() + 5 for matched in matched_all if len(matched) > 0])

    # Plot Wind Direction Histograms and Mole Fractions
    for i, (year, wd, matched) in enumerate(zip(config_years, wd_all, matched_all)):
        ax = axs[i]
        ax.hist(wd, bins=bins, weights=np.full(len(wd), 100 / max(len(wd), 1)),
                color='gray', edgecolor='white', label='Wind Direction')
        ax.set_xlim(0, 360)
        ax.set_ylim(0, hist_top)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_ylabel('Frequency (%)', fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs)

        # Snap N2O Points to the Center of their Wind Direction Bin
        ibin = np.clip(np.floor(matched['Wind Direction'] / config_bin_width), 0, len(bins) - 2)
        wd_center = ibin * config_bin_width + config_bin_width / 2

        ax2 = ax.twinx()
        ax2.scatter(wd_center, matched['N2O_C'], s=22, color=n2o_color,
                    edgecolors='black', linewidths=0.5, zorder=3, label=r'$\mathrm{N_2O}$ Observations')
        ax2.text(0.02, 0.87, year, transform=ax2.transAxes, color='black', fontsize=fs, fontweight='bold')
        ax2.set_ylim(335, n2o_top)
        ax2.set_ylabel(r'$\mathrm{N_2O}$ (ppb)', fontsize=fs, color=n2o_color)
        ax2.tick_params(axis='y', labelsize=fs, colors=n2o_color)
        ax2.spines['right'].set_color(n2o_color)

        # Formatting
        if i == len(axs) - 1:
            ax.set_xlabel('Wind Direction (°)', fontsize=fs)
        if i == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax2.legend(h1 + h2, l1 + l2, fontsize=fs-2, loc='upper right')

    # Add Subplot Labels
    for k, lab in enumerate(['(a)', '(b)', '(c)']):
        axs[k].text(-0.07, 1.00, lab, transform=axs[k].transAxes, fontsize=fs, fontweight='bold')

    plt.tight_layout()
    os.makedirs(config_fig_path, exist_ok=True)
    plt.savefig(f'{config_fig_path}fig04_windDirection.pdf', dpi=300)
    plt.close(fig)

# Main Execution
if __name__ == "__main__":
    dfObs2024, dfObs2025, dfObs2026 = load_observations(config_obs_path)

    met2024 = load_met(config_met_path, config_met_files['2024'])
    met2025 = load_met(config_met_path, config_met_files['2025'])
    met2026 = load_met(config_met_path, config_met_files['2026'])

    plot_wind_histograms([dfObs2024, dfObs2025, dfObs2026], [met2024, met2025, met2026])
