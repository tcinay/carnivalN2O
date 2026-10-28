# Plotting Script for Carnival N2O
# Timur Cinay, 2026
# The Script is designed to be called using plot.sh on with an HPC environment.
# WRF outputs are too large to include here. They can be made available upon request or run following the instructions in Table S1.

# Import Libraries
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import contextily as ctx

# Define Run Configurations
# Input Paths & File Formatting
config_wrf_path = '/home/gridsan/tcinay/wrf_gal/OUTPUTS/'   # edit to the local WRF output folder
config_wrf_runs = {'2024': 'CarnavalGAL2024_base', '2025': 'CarnavalGAL2025_base', '2026': 'CarnavalGAL2026_base'}
config_wrf_domain = 'd04'
config_stilt_path = './inputs/'
config_fig_path = './figures/'
config_basemap = './inputs/basemaps/sancristobal_osm_z13.tif'  # offline OpenStreetMap tile (no network access on HPC)
config_years = ['2024', '2025', '2026']

# Unit & Time Conversions
config_utc_offset = pd.Timedelta(hours=6)  # Galapagos is GMT-6

# Carnival Dates (Local, GMT-6)
config_carnival_dates = {
    '2024': ['2024-02-12', '2024-02-14'],
    '2025': ['2025-03-03', '2025-03-05'],
    '2026': ['2026-02-16', '2026-02-18'],
}

# Site Location and Urban Mask (keep in sync with inversion)
gal_lat, gal_lon = -0.89562, -89.60866
city_lat_min, city_lat_max = -0.921, -0.885
city_lon_min, city_lon_max = -89.622, -89.585
extent = [city_lon_min - 0.01, city_lon_max + 0.01, city_lat_min - 0.01, city_lat_max + 0.01]

# Figure Layout (all sizes in inches; figure size is derived from the panel size)
panel_in = 1.75                # side of each (near-square) map panel
left_in, right_in, top_in = 0.55, 0.10, 0.10
wgap_in = 1.0                  # middle column between the two maps, holds the row titles
hgap_in = 0.12                 # gap between rows (no tick labels in between)
xlab_in = 0.25                 # room for longitude labels under the bottom row
cbar_gap_in, cbar_h_in, cbar_lab_in = 0.15, 0.12, 0.60

# Load WRF
def load_wrf(year):
    ds = xr.open_mfdataset(f'{config_wrf_path}{config_wrf_runs[year]}/wrfout_{config_wrf_domain}*', combine='nested', concat_dim='Time')
    return ds.assign_coords(Time=ds['XTIME'])

# Load STILT Footprints, Convert to Local Time and Select the Carnival Window
def load_footprints(year, backtraj_folder):
    ds = xr.open_dataset(f'{backtraj_folder}stilt_GAL_WRF_{year}_baseWRF_every30min_1hrBack.nc')
    ds['time'] = ds['time'] - config_utc_offset
    d0, d1 = pd.to_datetime(config_carnival_dates[year])
    return ds.where((ds['time'] >= d0) & (ds['time'] <= d1), drop=True)

# Plotting Functions
# All panels share one extent, so lat labels go on the left column and lon labels on the bottom row only
def style_gridlines(ax, left_labels=True, bottom_labels=True):
    gl = ax.gridlines(draw_labels=True, alpha=0.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = left_labels
    gl.bottom_labels = bottom_labels
    gl.ylocator = mticker.MultipleLocator(0.02)
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

# Add the Offline Basemap and Hide the OpenStreetMap Attribution Text
def add_basemap(ax):
    #ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, zoom=13)
    ctx.add_basemap(ax, crs='EPSG:4326', source=config_basemap, reset_extent=False)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    for artist in ax.get_children():
        if hasattr(artist, 'get_text'):
            try:
                if 'OpenStreetMap' in str(artist.get_text()):
                    artist.set_visible(False)
            except TypeError:
                pass

def plot_wrf_maps(ax, ds, date_range, bottom_labels=True):
    # Select time period and get coordinates
    domain = ds.sel(Time=slice(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
    lats = domain['XLAT'].mean(dim='Time').values
    lons = domain['XLONG'].mean(dim='Time').values

    # Extract and average over full domain
    ws = np.sqrt(domain['U10']**2 + domain['V10']**2).mean(dim='Time').values
    u10 = domain['U10'].mean(dim='Time').values
    v10 = domain['V10'].mean(dim='Time').values

    # Plot
    levels_wrf = np.linspace(0, 4, 21)
    im = ax.contourf(lons, lats, ws, cmap='viridis', levels=levels_wrf, extend='max', alpha=0.5)
    ax.quiver(lons[::2, ::2], lats[::2, ::2], u10[::2, ::2], v10[::2, ::2], scale=50, width=0.004, zorder=5)
    ax.scatter(gal_lon, gal_lat, color='red', marker='*', s=40, zorder=6)
    style_gridlines(ax, left_labels=True, bottom_labels=bottom_labels)
    add_basemap(ax)

    return im

def plot_srr_maps(ax, dsFP, bottom_labels=True):
    style_gridlines(ax, left_labels=False, bottom_labels=bottom_labels)

    srr_mean = dsFP['srr'].mean(dim='time').values * 1000.0
    log_srr = np.where(srr_mean > 0, np.log(srr_mean), np.nan)
    longitudes, latitudes = np.meshgrid(dsFP['lon'].values, dsFP['lat'].values)

    levels_srr = np.linspace(1, 3.8, 11)
    im = ax.contourf(longitudes, latitudes, log_srr, levels=levels_srr, cmap='hot', alpha=0.7, extend='max', transform=ccrs.PlateCarree(), zorder=1)
    ax.scatter(gal_lon, gal_lat, color='red', marker='*', s=40, transform=ccrs.PlateCarree(), zorder=6)

    rect = Rectangle(
        (city_lon_min, city_lat_min),
        city_lon_max - city_lon_min,
        city_lat_max - city_lat_min,
        linewidth=1.5, edgecolor='blue', facecolor='none',
        transform=ccrs.PlateCarree(), zorder=5
    )
    ax.add_patch(rect)
    add_basemap(ax)

    return im

# Two-Line Row Label, e.g. '2024\nFeb 12–14'
def row_title(date_range):
    d0, d1 = pd.to_datetime(date_range)
    return f"{d0.year}\n{d0:%b} {d0.day}–{d1.day}"

def plot_maps(dsWRF_all, dsFP_all):
    # Figure Setup (explicit layout, so no tight_layout)
    fig_w = left_in + 2 * panel_in + wgap_in + right_in
    fig_h = top_in + 3 * panel_in + 2 * hgap_in + xlab_in + cbar_gap_in + cbar_h_in + cbar_lab_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    def add_ax(x_in, y_in, w_in, h_in, **kw):
        return fig.add_axes([x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h], **kw)

    x_col = [left_in, left_in + panel_in + wgap_in]
    y_bottom_row = cbar_lab_in + cbar_h_in + cbar_gap_in + xlab_in
    y_row = [y_bottom_row + (2 - r) * (panel_in + hgap_in) for r in range(3)]

    proj = ccrs.PlateCarree()
    axs_wrf = [add_ax(x_col[0], y_row[r], panel_in, panel_in, projection=proj) for r in range(3)]
    axs_srr = [add_ax(x_col[1], y_row[r], panel_in, panel_in, projection=proj) for r in range(3)]
    cax1 = add_ax(x_col[0], cbar_lab_in, panel_in, cbar_h_in)
    cax2 = add_ax(x_col[1], cbar_lab_in, panel_in, cbar_h_in)

    # Plot WRF and SRR Maps
    for r, year in enumerate(config_years):
        bottom = (r == len(config_years) - 1)
        im_wrf = plot_wrf_maps(axs_wrf[r], dsWRF_all[r], config_carnival_dates[year], bottom_labels=bottom)
        im_srr = plot_srr_maps(axs_srr[r], dsFP_all[r], bottom_labels=bottom)

    fig.colorbar(im_wrf, cax=cax1, orientation='horizontal', label=r'10-m Wind Speed (m s$^{-1}$)')
    fig.colorbar(im_srr, cax=cax2, orientation='horizontal', label='STILT Footprints\n' r'($\mathrm{log[ppb \: (\mu mol^{-1} \: m^2 \: s)]}$)')

    # Panel Labels
    panel_axs = [ax for pair in zip(axs_wrf, axs_srr) for ax in pair]
    for ax, label in zip(panel_axs, ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
        ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=8, fontweight='bold', va='top', ha='left', zorder=10)

    # Row Titles: Centered in the Middle Column, Vertically Centered on Each Row
    x_text = (x_col[0] + panel_in + wgap_in / 2) / fig_w
    for r, year in enumerate(config_years):
        y_text = (y_row[r] + panel_in / 2) / fig_h
        fig.text(x_text, y_text, row_title(config_carnival_dates[year]), ha='center', va='center',
                 fontsize=9, fontweight='bold', linespacing=1.3)

    # Save Figure
    os.makedirs(config_fig_path, exist_ok=True)
    plt.savefig(f'{config_fig_path}fig05_maps.pdf', dpi=300)
    plt.close(fig)

# Main Execution
if __name__ == "__main__":
    dsWRF_all = [load_wrf(year) for year in config_years]
    dsFP_all  = [load_footprints(year, config_stilt_path) for year in config_years]
    plot_maps(dsWRF_all, dsFP_all)
