# Forward Model Plotting Script for Carnival N2O
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
config_stilt_path = './inputs/'
config_obs_path = './inputs/'
config_fig_path = './figures/'
config_year = 2026

# Unit & Time Conversions
config_utc_offset = pd.Timedelta(hours=6)                 # Galapagos is GMT-6
config_conv_umol_s_to_g_hr = 1e-6 * 44.013 * 3600         # same factor as inversion

# Background Method (see estimate_background)
config_bg_method = 'median'

# Emission Basis
# 'per_cell' : rate multiplies the summed city sensitivity, identical to ypred in the inversion
# 'total'    : rate is the whole-city total, spread evenly across city mask cells
config_emiss_basis = 'per_cell'

# Carnival Evenings (Local Date of each Emission Pulse)
config_event_nights = ['2026-02-16']

# Log-Normal Emission Pulse (Local Time, g/hr)
# Shape fit by eye to the 2024 STILT-BayesianMCMC posterior: onset 17:00, peak 18:30, tail to ~22:00
config_pulse = {
    'onset':     '17:00',   # emissions are zero before this time
    'peak_rate': 1500.0,    # g/hr at the mode
    'peak_hr':   1.5,       # hours after onset at which the pulse peaks
    'sigma':     0.55,      # log-normal shape parameter (larger = longer tail)
}

# Load STILT Footprints
def load_footprints(year, backtraj_folder):
    fpath = f'{backtraj_folder}stilt_GAL_WRF_{year}_baseWRF_every30min_1hrBack.nc'

    with xr.open_dataset(fpath) as dsFP:
        times     = pd.to_datetime(dsFP.time.values)
        srr_raw   = dsFP['srr'].values
        grid_area = dsFP['grid_area'].values
        lats      = dsFP['lat'].values
        lons      = dsFP['lon'].values

    sensitivity = srr_raw * 1000 / grid_area[np.newaxis, :, :] # unit conversion, as in inversion

    return times, sensitivity, lats, lons

# Load Observations (Data S1, one sheet per year, UTC)
def load_observations(folder_path, year, filename='DataS1.xlsx', to_local=True):
    df = pd.read_excel(folder_path + filename, sheet_name=str(year))
    df = df[['datetime', 'N2O_C']].copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    if to_local:
        df['datetime'] = df['datetime'] - config_utc_offset
    return df.sort_values('datetime').reset_index(drop=True)

# Average Observations in 30-min Windows Centered on Footprint Times (both in local time)
def bin_observations(dfObs, times_local):
    dfObs = dfObs.dropna(subset=['N2O_C'])
    half = pd.Timedelta(minutes=15)
    yobs = np.full(len(times_local), np.nan)
    for k, t in enumerate(times_local):
        window = dfObs[(dfObs['datetime'] >= t - half) & (dfObs['datetime'] < t + half)]['N2O_C']
        if window.size > 0:
            yobs[k] = window.mean()
    return yobs

# Urban Mask Development (keep in sync with inversion)
def define_city_mask(srr, lats, lons):
    lat_min = -0.921
    lat_max = -0.885
    lon_min = -89.622
    lon_max = -89.585

    city_mask = np.zeros(srr.shape[1:], dtype=bool)
    city_mask[np.ix_((lats >= lat_min) & (lats <= lat_max), (lons >= lon_min) & (lons <= lon_max))] = True

    return city_mask

# Estimate Background Mole Fractions
def estimate_background(times, yobs, method='median'):
    finite = yobs[np.isfinite(yobs)]

    if method == 'median':
        return float(np.median(finite))

    if method == 'p10':
        return float(np.percentile(finite, 10))

    if method == 'p25':
        return float(np.percentile(finite, 25))

    if method == 'rolling_6hr':
        s = pd.Series(yobs, index=pd.DatetimeIndex(times))
        roll = s.rolling('6h', center=True, min_periods=2).quantile(0.10)
        vals = roll.interpolate(limit_direction='both').to_numpy(dtype=float)
        vals = np.where(np.isfinite(vals), vals, float(np.median(finite)))
        return vals

    raise ValueError(f"Unknown background method '{method}'")

# Log-Normal Pulse Shape, Normalized to 1 at the Peak
def lognormal_shape(tau_hr, peak_hr, sigma):
    mu = np.log(peak_hr) + sigma**2  # mode of a log-normal is exp(mu - sigma^2)
    shape = np.zeros_like(tau_hr, dtype=float)
    pos = tau_hr > 0
    shape[pos] = (peak_hr / tau_hr[pos]) * np.exp(-((np.log(tau_hr[pos]) - mu)**2 - (np.log(peak_hr) - mu)**2) / (2 * sigma**2))
    return shape

# Build Prescribed Emission Time Series (g/hr) on the Footprint Time Axis
def build_emissions(times_local, pulse, nights):
    emiss = np.zeros(len(times_local))

    for night in pd.to_datetime(nights):
        t0 = night + pd.Timedelta(pulse['onset'] + ':00')
        tau_hr = np.asarray((times_local - t0).total_seconds(), dtype=float) / 3600
        emiss += pulse['peak_rate'] * lognormal_shape(tau_hr, pulse['peak_hr'], pulse['sigma'])

    return emiss

# Forward Model: ypred = bg + emiss_city * city_srr
def forward_model(sensitivity, city_mask, emiss_g_hr, ybg, basis='per_cell'):
    city_srr = np.nansum(sensitivity * city_mask, axis=(1, 2))

    emiss_umol_s = emiss_g_hr / config_conv_umol_s_to_g_hr
    if basis == 'total':
        emiss_umol_s = emiss_umol_s / city_mask.sum()

    enhancement = emiss_umol_s * city_srr
    ybg_arr = np.broadcast_to(np.asarray(ybg, dtype=float), enhancement.shape)

    return ybg_arr + enhancement, enhancement

# Print Statistics
def print_forward_stats(times_local, emiss, ypred):
    k_emiss = int(np.argmax(emiss))
    k = int(np.nanargmax(ypred))

    print(f"Emission peak: {emiss[k_emiss]:.0f} g/hr at {times_local[k_emiss]} (GMT-6)")
    print(f"Peak forward-modeled N2O: {ypred[k]:.1f} ppb at {times_local[k]} (GMT-6)")

# Plotting Function
def plot_forward_model(dfObs, times_local, emiss, ypred):
    fig, axs = plt.subplots(2, 1, figsize=(7, 6))
    fs = 12

    # Carnival Dates
    dates = pd.to_datetime(['2026-02-16', '2026-02-18'])

    # Prescribed Emissions
    ax = axs[0]
    ax.plot(times_local, emiss, color='royalblue', linestyle='-', label='Log-Normal Pulse')
    ax.text(dates[0] + pd.Timedelta(hours=2), ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 150,
            str(config_year), color='black', fontsize=fs, fontweight='bold')
    ax.set_xlim(dates[0], dates[1])
    ax.set_ylabel(r'$\mathrm{N_2O}$ Emissions (g hr$^{-1}$)', fontsize=fs)

    # Observed and Forward-Modeled Mole Fractions
    ax = axs[1]
    ax.plot(dfObs['datetime'], dfObs['N2O_C'], linestyle='-', color='black', label='Observations')
    ax.plot(times_local, ypred, color='royalblue', linestyle='-', label='Forward-Modeled')
    ax.set_xlim(dates[0], dates[1])
    ax.set_ylim(335, max(410, np.nanmax(ypred) + 5))
    ax.set_ylabel(r'$\mathrm{N_2O}$ (ppb)', fontsize=fs)
    ax.legend(fontsize=fs-2, loc='upper right')

    # Shared Formatting
    for ax in axs:
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
    plt.savefig(f'{config_fig_path}fig07_forwardModel.pdf', dpi=300)
    plt.close(fig)

# Main Execution
if __name__ == "__main__":
    # Load Observations (converted to local time)
    dfObs = load_observations(config_obs_path, config_year)

    # Load Footprints and Convert to Local Time
    times, sensitivity, lats, lons = load_footprints(config_year, config_stilt_path)
    times_local = times - config_utc_offset
    city_mask = define_city_mask(sensitivity, lats, lons)

    # Background from 30-min Observation Means
    yobs = bin_observations(dfObs, times_local)
    ybg = estimate_background(times_local, yobs, method=config_bg_method)

    # Forward Model with Log-Normal Emission Pulse
    emiss = build_emissions(times_local, config_pulse, config_event_nights)
    ypred, enhancement = forward_model(sensitivity, city_mask, emiss, ybg, basis=config_emiss_basis)
    print_forward_stats(times_local, emiss, ypred)
    plot_forward_model(dfObs, times_local, emiss, ypred)
