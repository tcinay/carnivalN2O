# Plotting Script for Carnival N2O 
# Timur Cinay, 2026
# The Script is designed to be called using plot.sh on with an HPC environment.

# Import Libraries 
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load STILT-Inversion Results
def load_inversion_results(folder_path):
    ds2024 = xr.open_dataset(folder_path + 'inversionResults_2024.nc')
    ds2025 = xr.open_dataset(folder_path + 'inversionResults_2025.nc')
    return ds2024, ds2025


# Plot Emissions
def plot_emissions(ds2024, ds2025):

    fig, axs = plt.subplots(2, 1, figsize=(7, 6))
    fs = 12

    # Carnival Dates
    dates2024 = pd.to_datetime(['2024-02-12', '2024-02-14'])
    dates2025 = pd.to_datetime(['2025-03-03', '2025-03-05'])

    # Conversion factor: µmol s-1 -> g hr-1
    conv = 1e-6 * 44.013 * 3600

    # 2024
    ax = axs[0]
    post2024 = ds2024['emiss_city_mean'] * conv
    err2024  = ds2024['emiss_city_std']  * conv
    ax.plot(ds2024['time'] - pd.Timedelta('6h'), post2024, color='royalblue', linestyle='-', label='STILT-BayesianMCMC')
    ax.fill_between(ds2024['time'] - pd.Timedelta('6h'), post2024 - err2024, post2024 + err2024, color='royalblue', alpha=0.3, linewidth=0)
    ax.text(dates2024[0] + pd.Timedelta(hours=2), ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 150,
            '2024', color='black', fontsize=fs, fontweight='bold')
    ax.set_xlim(dates2024[0], dates2024[1])

    # 2025
    ax = axs[1]
    post2025 = ds2025['emiss_city_mean'] * conv
    err2025  = ds2025['emiss_city_std']  * conv
    ax.plot(ds2025['time'] - pd.Timedelta('6h'), post2025, color='royalblue', linestyle='-', label='STILT-BayesianMCMC')
    ax.fill_between(ds2025['time'] - pd.Timedelta('6h'), post2025 - err2025, post2025 + err2025, color='royalblue', alpha=0.3, linewidth=0)
    ax.text(dates2025[0] + pd.Timedelta(hours=2), ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 150,
            '2025', color='black', fontsize=fs, fontweight='bold')
    ax.set_xlim(dates2025[0], dates2025[1])

    # Formatting 
    for ax in axs:
        ax.set_ylabel(r'$\mathrm{N_2O}$ Emissions (g hr$^{-1}$)', fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs)
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[6, 18]))
        ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
   
    axs[-1].set_xlabel('Time (GMT-6)', fontsize=fs)
    axs[0].legend(fontsize=fs)

    # Panel labels
    axs[0].text(-0.2, 1.00, '(a)', transform=axs[0].transAxes, fontsize=fs, fontweight='bold')
    axs[1].text(-0.2, 1.00, '(b)', transform=axs[1].transAxes, fontsize=fs, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/fig04_emissions.pdf', dpi=300)

# Print Stats
def print_emission_stats(ds2024, ds2025):
    conv = 1e-6 * 44.013 * 3600
    print("=" * 60)
    print("STILT Inversion")
    print("=" * 60)
    for year, ds in [('2024', ds2024), ('2025', ds2025)]:
        city     = ds['emiss_city_mean']
        city_err = ds['emiss_city_std']
        nc       = ds['emiss_noncity_mean']
        nc_err   = ds['emiss_noncity_std']
        bg       = ds['bg_mean']
        bg_err   = ds['bg_std']

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

        peak_pred_enh = float((ds['ypred_mean'] - ds['ybg']).max())
        peak_obs_enh  = float((ds['yobs'] - ds['ybg']).max())
        prior_q       = float(ds['prior_emiss']) * conv

        print(f"  {year}:")
        print(f"    prior city Q      : {prior_q:.1f} g/hr")
        print(f"    peak city Q       : {peak_q:.1f} ± {peak_qerr:.1f} g/hr  (1-sigma)")
        print(f"    mean active city Q: {mean_q:.1f} ± {mean_qerr:.1f} g/hr")
        print(f"    mean non-city Q   : {mean_nc:.1f} ± {mean_ncerr:.1f} g/hr")
        print(f"    background (ppb)  : {mean_bg:.3f} ± {mean_bgerr:.3f} ppb")
        print(f"    peak pred enh     : {peak_pred_enh:.2f} ppb")
        print(f"    peak obs  enh     : {peak_obs_enh:.2f} ppb")
 
# Execute
if __name__ == "__main__":
    folder_path = 'outputs/'
    ds2024, ds2025 = load_inversion_results(folder_path)
    plot_emissions(ds2024, ds2025)
    
    print_emission_stats(ds2024, ds2025)