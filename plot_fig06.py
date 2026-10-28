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

# Define Run Configurations
# Input Paths & File Formatting
config_inv_path = './outputs/'
config_fig_path = './figures/'

# Unit & Time Conversions
config_utc_offset = pd.Timedelta(hours=6)                  # Galapagos is GMT-6
config_conv_umol_s_to_g_hr = 1e-6 * 44.013 * 3600          # same factor as inversion

# Carnival Dates (Local, GMT-6)
config_carnival_dates = {
    '2024': pd.to_datetime(['2024-02-12', '2024-02-14']),
    '2025': pd.to_datetime(['2025-03-03', '2025-03-05']),
}

# Load STILT-Inversion Results
def load_inversion_results(folder_path):
    ds2024 = xr.open_dataset(folder_path + 'inversionResults_2024.nc')
    ds2025 = xr.open_dataset(folder_path + 'inversionResults_2025.nc')

    ds2024['time'] = ds2024['time'] - config_utc_offset
    ds2025['time'] = ds2025['time'] - config_utc_offset

    return ds2024, ds2025

# Plot Emissions
def plot_emissions(ds_all):
    fig, axs = plt.subplots(2, 1, figsize=(7, 6))
    fs = 12
    conv = config_conv_umol_s_to_g_hr

    for ax, (year, ds) in zip(axs, ds_all.items()):
        dates = config_carnival_dates[year]
        post = ds['emiss_city_mean'] * conv
        err  = ds['emiss_city_std']  * conv

        # Posterior Mean +/- 1 Sigma
        ax.plot(ds['time'], post, color='royalblue', linestyle='-', label='STILT-BayesianMCMC')
        ax.fill_between(ds['time'], post - err, post + err, color='royalblue', alpha=0.3, linewidth=0)
        ax.text(dates[0] + pd.Timedelta(hours=2), ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 150,
                year, color='black', fontsize=fs, fontweight='bold')
        ax.set_xlim(dates[0], dates[1])

    # Formatting
    for ax in axs:
        ax.set_ylabel(r'$\mathrm{N_2O}$ Emissions (g hr$^{-1}$)', fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[6, 18]))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))

    axs[-1].set_xlabel('Time (GMT-6)', fontsize=fs)

    # Panel Labels
    axs[0].text(-0.2, 1.00, '(a)', transform=axs[0].transAxes, fontsize=fs, fontweight='bold')
    axs[1].text(-0.2, 1.00, '(b)', transform=axs[1].transAxes, fontsize=fs, fontweight='bold')
    plt.tight_layout()
    os.makedirs(config_fig_path, exist_ok=True)
    plt.savefig(f'{config_fig_path}fig06_emissions.pdf', dpi=300)
    plt.close(fig)

# Print Stats
def print_emission_stats(ds_all):
    conv = config_conv_umol_s_to_g_hr
    print("=" * 60)
    print("STILT Inversion")
    print("=" * 60)
    for year, ds in ds_all.items():
        city     = ds['emiss_city_mean']
        city_err = ds['emiss_city_std']
        nc       = ds['emiss_noncity_mean']
        nc_err   = ds['emiss_noncity_std']
        bg       = ds['bg_mean']
        bg_err   = ds['bg_std']

        # Timesteps with no valid obs are all-NaN, so use nan-aware argmax and mask on pulse_index
        peak_idx  = int(np.nanargmax(city.values))
        peak_q    = float(city[peak_idx]) * conv
        peak_qerr = float(city_err[peak_idx]) * conv
        active    = ds['pulse_index'].astype(bool)

        mean_q    = float(city.where(active).mean()) * conv if bool(active.any()) else 0.0
        mean_qerr = float(city_err.where(active).mean()) * conv if bool(active.any()) else 0.0

        mean_nc    = float(nc.mean()) * conv
        mean_ncerr = float(nc_err.mean()) * conv

        mean_bg    = float(bg.mean())
        mean_bgerr = float(bg_err.mean())

        # Total emissions over the enhancement period: mean active city Q x active duration
        dt_hr     = pd.Series(ds['time'].values).diff().median().total_seconds() / 3600
        active_hr = int(active.sum()) * dt_hr
        total_q    = mean_q * active_hr
        total_qerr = mean_qerr * active_hr

        peak_pred_enh = float((ds['ypred_mean'] - ds['ybg']).max())

        print(f"  {year}:")
        print(f"    peak city Q       : {peak_q:.1f} ± {peak_qerr:.1f} g/hr  (1-sigma)")
        print(f"    mean non-city Q   : {mean_nc:.1f} ± {mean_ncerr:.1f} g/hr")
        print(f"    mean active city Q: {mean_q:.1f} ± {mean_qerr:.1f} g/hr")
        print(f"    active duration   : {active_hr:.1f} hr")
        print(f"    total city Q      : {total_q:.0f} ± {total_qerr:.0f} g")
        print(f"    background (ppb)  : {mean_bg:.2f} ± {mean_bgerr:.2f} ppb")
        print(f"    peak pred enh     : {peak_pred_enh:.2f} ppb")

# Main Execution
if __name__ == "__main__":
    ds2024, ds2025 = load_inversion_results(config_inv_path)
    ds_all = {'2024': ds2024, '2025': ds2025}

    plot_emissions(ds_all)
    print_emission_stats(ds_all)
