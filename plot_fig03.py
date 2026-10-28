# Plotting Script for Carnival N2O
# Timur Cinay, 2026
# The Script is designed to be called using plot.sh on with an HPC environment.

# Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define Run Configurations
# Input Paths & File Formatting
config_obs_path = './inputs/'
config_fig_path = './figures/'
config_years = ['2024', '2025', '2026']

# Unit & Time Conversions
config_utc_offset = pd.Timedelta(hours=6)  # Galapagos is GMT-6

# Carnival Dates (Local, GMT-6) and the Pre/Post-Carnival Window Length
config_carnival_dates = {
    '2024': pd.to_datetime(['2024-02-12', '2024-02-14']),
    '2025': pd.to_datetime(['2025-03-03', '2025-03-05']),
    '2026': pd.to_datetime(['2026-02-16', '2026-02-18']),
}
config_pad = pd.Timedelta(days=2)

# Evening Window (Local Hour, End Exclusive); Evening Samples are Filled, Others Open
config_hour_start = 16
config_hour_end = 24

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

# Select Evening Samples (config_hour_start to config_hour_end, Local)
def is_evening(df):
    h = df['datetime'].dt.hour
    return (h >= config_hour_start) & (h < config_hour_end)

# Plotting Functions
# Evening Samples Filled, All Other Hours Open (Unfilled)
def scatter_split(ax, df, color, label, s=10, lw=0.8):
    ev = is_evening(df)
    d_ev, d_day = df[ev], df[~ev]
    ax.scatter(d_ev['N2O_C'], d_ev['CO_C'], s=s, label=label, facecolors=color, edgecolors=color, linewidths=lw)
    ax.scatter(d_day['N2O_C'], d_day['CO_C'], s=s, facecolors='none', edgecolors=color, linewidths=lw)

def plot_comparison_COvN2O(dfObs_all):
    fig, axs = plt.subplots(3, 1, figsize=(4.5, 6))
    fs = 12

    for ax, year, df in zip(axs, config_years, dfObs_all):
        c0, c1 = config_carnival_dates[year]
        t = df['datetime']
        df_carnival     = df[(t >= c0) & (t <= c1)]
        df_precarnival  = df[(t < c0) & (t >= c0 - config_pad)]
        df_postcarnival = df[(t > c1) & (t <= c1 + config_pad)]

        scatter_split(ax, df_carnival,     'lightblue', 'Carnival')
        scatter_split(ax, df_precarnival,  'gray',      'Pre-Carnival')
        scatter_split(ax, df_postcarnival, 'black',     'Post-Carnival')

        ax.set_xlabel(r'$\mathrm{N_2O}$ (ppb)', fontsize=fs)
        ax.set_ylabel(r'$\mathrm{CO}$ (ppb)', fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs)
        ax.text(0.97, 0.95, year, transform=ax.transAxes, ha='right', va='top', fontsize=fs, fontweight='bold')

    axs[2].legend(loc='upper left', fontsize=fs-4)

    # Add Subplot Labels
    for k, lab in enumerate(['(a)', '(b)', '(c)']):
        axs[k].text(-0.1, 1.00, lab, transform=axs[k].transAxes, fontsize=fs, fontweight='bold')

    plt.tight_layout()
    os.makedirs(config_fig_path, exist_ok=True)
    plt.savefig(f'{config_fig_path}fig03_COvN2O.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

# Main Execution
if __name__ == "__main__":
    dfObs2024, dfObs2025, dfObs2026 = load_observations(config_obs_path)
    plot_comparison_COvN2O([dfObs2024, dfObs2025, dfObs2026])
