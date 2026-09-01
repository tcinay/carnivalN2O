# Plotting Script for Carnival N2O 
# Timur Cinay, 2026
# The Script is designed to be called using plot.sh on with an HPC environment.

# Load Libraries
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import contextily as ctx

# Load WRF 
dsWRF2024 = xr.open_mfdataset('/home/gridsan/tcinay/wrf_gal/OUTPUTS/CarnavalGAL2024_base/wrfout_d04*', combine='nested', concat_dim='Time')
dsWRF2024 = dsWRF2024.assign_coords(Time=dsWRF2024['XTIME'])
dsWRF2025 = xr.open_mfdataset('/home/gridsan/tcinay/wrf_gal/OUTPUTS/CarnavalGAL2025_base/wrfout_d04*', combine='nested', concat_dim='Time') 
dsWRF2025 = dsWRF2025.assign_coords(Time=dsWRF2025['XTIME'])
dsWRF2026 = xr.open_mfdataset('/home/gridsan/tcinay/wrf_gal/OUTPUTS/CarnavalGAL2026_base/wrfout_d04*', combine='nested', concat_dim='Time')
dsWRF2026 = dsWRF2026.assign_coords(Time=dsWRF2026['XTIME'])

# Load SRR
dsFP2024 = xr.open_dataset(f'./inputs/stilt_GAL_WRF_2024_baseWRF_every30min_1hrBack.nc')
dsFP2025 = xr.open_dataset(f'./inputs/stilt_GAL_WRF_2025_baseWRF_every30min_1hrBack.nc')
dsFP2026 = xr.open_dataset(f'./inputs/stilt_GAL_WRF_2026_baseWRF_every30min_1hrBack.nc')

# Time Filter the SRR
dsFP2024['time'] = dsFP2024['time'] - pd.Timedelta('6h')
dsFP2025['time'] = dsFP2025['time'] - pd.Timedelta('6h')
dsFP2026['time'] = dsFP2026['time'] - pd.Timedelta('6h')

dsFP2024 = dsFP2024.where((dsFP2024['time'] >= pd.to_datetime('2024-02-12')) & (dsFP2024['time'] <= pd.to_datetime('2024-02-14')), drop=True)
dsFP2025 = dsFP2025.where((dsFP2025['time'] >= pd.to_datetime('2025-03-03')) & (dsFP2025['time'] <= pd.to_datetime('2025-03-05')), drop=True)
dsFP2026 = dsFP2026.where((dsFP2026['time'] >= pd.to_datetime('2026-02-16')) & (dsFP2026['time'] <= pd.to_datetime('2026-02-18')), drop=True)

# Shared constants
gal_lat, gal_lon = -0.89562, -89.60866
city_lat_min, city_lat_max = -0.921, -0.885
city_lon_min, city_lon_max = -89.622, -89.585
extent = [city_lon_min - 0.01, city_lon_max + 0.01, city_lat_min - 0.01, city_lat_max + 0.01]
basemap = './inputs/basemaps/sancristobal_osm_z13.tif' # only use if no network access available on HPC

# Figure Setup 
fig = plt.figure(figsize=(6.5, 6))
gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.1])
ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax4 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax5 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
ax7 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
ax8 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
cax1 = fig.add_subplot(gs[3, 0])
cax2 = fig.add_subplot(gs[3, 1])

# Plotting Functions
def plot_wrf_maps(ax, ds, date_range):
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
    gl = ax.gridlines(draw_labels=True, alpha=0.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    #ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, zoom=13)
    ctx.add_basemap(ax, crs='EPSG:4326', source=basemap, reset_extent=False)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    for artist in ax.get_children():
        if hasattr(artist, 'get_text'):
            try:
                if 'OpenStreetMap' in str(artist.get_text()):
                    artist.set_visible(False)
            except TypeError:
                pass

    return im

# Plot WRF 
im1 = plot_wrf_maps(ax1, dsWRF2024, ['2024-02-12', '2024-02-14'])
im2 = plot_wrf_maps(ax4, dsWRF2025, ['2025-03-03', '2025-03-05'])
im3 = plot_wrf_maps(ax7, dsWRF2026, ['2026-02-16', '2026-02-18'])

fig.colorbar(im1, cax=cax1, orientation='horizontal', label=r'10-m Wind Speed (m s$^{-1}$)', shrink=0.6)
titles = '2024-02-12 to 2024-02-14', '2025-03-03 to 2025-03-05', '2026-02-16 to 2026-02-18'

for ax, title in zip([ax1, ax4, ax7], titles):
    ax.set_title(title, fontsize=8, fontweight='bold', pad=5)

# Plot SRR Maps
levels_srr = np.linspace(1, 3.8, 11)

for ax, dsFPx, yr in zip([ax2, ax5, ax8], [dsFP2024, dsFP2025, dsFP2026], [2024, 2025, 2026]):
    gl = ax.gridlines(draw_labels=True, alpha=0.2)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 8}
    gl.ylabel_style = {'size': 8}

    srr_mean = dsFPx['srr'].mean(dim='time').values * 1000.0
    log_srr = np.where(srr_mean > 0, np.log(srr_mean), np.nan)
    longitudes, latitudes = np.meshgrid(dsFPx['lon'].values, dsFPx['lat'].values)

    im_srr = ax.contourf(longitudes, latitudes, log_srr, levels=levels_srr, cmap='hot', alpha=0.7, extend='max', transform=ccrs.PlateCarree(), zorder=1)
    ax.scatter(gal_lon, gal_lat, color='red', marker='*', s=40, transform=ccrs.PlateCarree(), zorder=6)

    rect = Rectangle(
        (city_lon_min, city_lat_min),
        city_lon_max - city_lon_min,
        city_lat_max - city_lat_min,
        linewidth=1.5, edgecolor='blue', facecolor='none',
        transform=ccrs.PlateCarree(), zorder=5
    )
    ax.add_patch(rect)

    ctx.add_basemap(ax, crs='EPSG:4326', source=basemap, reset_extent=False)
    #ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, zoom=13)

    ax.set_extent(extent, crs=ccrs.PlateCarree())
    for artist in ax.get_children():
        if hasattr(artist, 'get_text'):
            try:
                if 'OpenStreetMap' in str(artist.get_text()):
                    artist.set_visible(False)
            except TypeError:
                pass

fig.colorbar(im_srr, cax=cax2, orientation='horizontal', label='STILT Footprints ' r'($\mathrm{log[ppb \: (\mu mol^{-1} \: m^2 \: s)]}$)')

# Panel labels
for ax, label in zip([ax1, ax2, ax4, ax5, ax7, ax8], ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
    ax.text(0.03, 0.97, label, transform=ax.transAxes, fontsize=8, fontweight='bold', va='top', ha='left', zorder=10)

# Save Figure
plt.tight_layout()
plt.savefig('figures/fig03_maps.pdf', dpi=300)


