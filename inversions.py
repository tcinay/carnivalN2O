# Inversion Script for Carnival N2O 
# Timur Cinay, 2026
# The Script is designed to be called using runInversion.sh on with an HPC environment.

# Import Libraries
import os
import warnings
import argparse
import numpy as np
import xarray as xr
import pandas as pd
import pymc as pm
import arviz as az

# Define Run Configurations
# Input Paths & File Formatting
config_stilt_path = "./inputs/"
config_obs_path = "./inputs/"

# PyMC Configs for Convergence Diagnostics
config_rhat_treshold = 1.05
config_rhat_fail = 1.1
config_ess_warning = 100

# PyMC Configs for Sampling
config_nDraws = 1000
config_nTune = 1000
config_nChains = 4

# Parse Arguments from Command Line / SLURM Job Submission
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=[2024, 2025, 2026], required=True)
    parser.add_argument("--cores", type=int, default=None)
    parser.add_argument( "--outdir", type=str, default="outputs")
    return parser.parse_args()

# Load and Preprocess Inputs
def preprocess_inputs(year, obs_folder, backtraj_folder):
    # Load STILT Footprints
    fpath = (f"{backtraj_folder}/stilt_GAL_WRF_{year}_baseWRF_every30min_1hrBack.nc")

    with xr.open_dataset(fpath) as dsFP:
        times     = pd.to_datetime(dsFP.time.values)
        srr_raw   = dsFP["srr"].values
        grid_area = dsFP["grid_area"].values
        lats      = dsFP["lat"].values
        lons      = dsFP["lon"].values

    sensitivity = srr_raw * 1000 / grid_area[np.newaxis, :, :] # unit conversion 

    # Load Observations
    dfObs = pd.read_excel(f"{obs_folder}/DataS1.xlsx", sheet_name=str(year))
    dfObs["datetime"] = pd.to_datetime(dfObs["datetime"])
    dfObs = dfObs[["datetime", "N2O_C"]].dropna(subset=["N2O_C"]).sort_values("datetime").reset_index(drop=True)

    # Calculate Observed Means and Standard Deviations in 30-min Windows
    y_means, y_stds, y_counts = [], [], []
    half = pd.Timedelta(minutes=15)
    for t in times:
        window = dfObs[(dfObs["datetime"] >= t - half) & (dfObs["datetime"] < t + half)]["N2O_C"]
        n = window.size
        y_counts.append(n)
        y_means.append(window.mean() if n > 0 else np.nan)
        y_stds.append(window.std() if n > 1 else np.nan)

    yobs = np.asarray(y_means, dtype=float)
    ystd = np.asarray(y_stds, dtype=float)
    nobs = np.asarray(y_counts, dtype=int)

    with np.errstate(invalid="ignore", divide="ignore"):
        ysem = ystd / np.sqrt(np.where(nobs > 0, nobs, np.nan)) # standard error

    return times, yobs, ystd, ysem, nobs, sensitivity, lats, lons

# Urban Mask Development
def define_city_mask(srr, lats, lons):
    # Edit for Sensitivity
    lat_min = -0.921 
    lat_max= -0.885
    lon_min = -89.622
    lon_max = -89.585

    city_mask = np.zeros(srr.shape[1:], dtype=bool)
    city_mask[np.ix_((lats >= lat_min) & (lats <= lat_max), (lons >= lon_min) & (lons <= lon_max))] = True

    return city_mask

# Estimate Background Mole Fractions
def estimate_background(times, yobs, method="median"):
    finite = yobs[np.isfinite(yobs)]

    if method == "median":
        return float(np.median(finite))
    
    if method == "p10":
        return float(np.percentile(finite, 10))
    
    if method == "p25":
        return float(np.percentile(finite, 25))
    
    if method == "rolling_6hr":
        s = pd.Series(yobs, index=pd.DatetimeIndex(times))
        roll = (s.rolling("6h", center=True, min_periods=2).quantile(10))
        vals = roll.interpolate(limit_direction="both").to_numpy(dtype=float)
        vals = np.where(np.isfinite(vals), vals, float(np.median(finite)))
        return vals

# Estimate Prior Emissions
def get_prior_emissions(yobs, ystd, srr, city_mask, ybg, prior_norm="city", prior_emiss_scale=1.0, emiss_noncity_scale=1e-3): 

    ybg_arr = np.broadcast_to(np.asarray(ybg, dtype=float), yobs.shape)
    with np.errstate(invalid="ignore"):
        pulse_index = (yobs - ybg_arr) > 2.0 * np.nanmean(ystd)
    pulse_index = np.asarray(pulse_index, dtype=bool)

    basis = city_mask if prior_norm == "city" else np.ones_like(city_mask, dtype=bool)

    mean_fp_pulse    = np.nanmean(srr[pulse_index], axis=0)
    mean_enhancement = np.nanmean((yobs - ybg_arr)[pulse_index])
    fp_norm          = np.nansum(mean_fp_pulse * basis)

    prior_emiss         = prior_emiss_scale * mean_enhancement / fp_norm
    prior_emiss_noncity = emiss_noncity_scale * prior_emiss


    city_srr_all    = np.nansum(srr *  city_mask, axis=(1, 2))
    noncity_srr_all = np.nansum(srr * ~city_mask, axis=(1, 2))
    ypred_prior = ybg_arr + (prior_emiss * city_srr_all + prior_emiss_noncity * noncity_srr_all) * pulse_index.astype(float)

    return prior_emiss, prior_emiss_noncity, ypred_prior, pulse_index

# PyMC Check for Convergence Diagnostics
def check_convergence(idata):
    rhat = az.rhat(idata)
    ess  = az.ess(idata)

    max_rhat, min_ess = np.nan, np.nan

    for var in rhat.data_vars:
        r = float(np.nanmax(rhat[var].values))
        e = float(np.nanmin(ess[var].values))
        max_rhat = r if not np.isfinite(max_rhat) else max(max_rhat, r)
        min_ess  = e if not np.isfinite(min_ess)  else min(min_ess, e)

        if r > config_rhat_fail:
            warnings.warn(f"max R-hat={r:.3f} > {config_rhat_fail} for '{var}' — poor convergence.", stacklevel=2,)
        elif r > config_rhat_treshold:
            warnings.warn(f"max R-hat={r:.3f} > {config_rhat_treshold} for '{var}' — marginal convergence.", stacklevel=2)
        if e < config_ess_warning:
            warnings.warn(f"min ESS={e:.0f} < {config_ess_warning} for '{var}' — consider more draws.", stacklevel=2)

    ndiv = 0
    if "diverging" in idata.sample_stats:
        ndiv = int(idata.sample_stats["diverging"].values.sum())
        if ndiv > 0:
            warnings.warn(f"{ndiv} divergent transitions — consider raising target_accept.", stacklevel=2)

    return {"max_rhat":    float(max_rhat), "min_ess":     float(min_ess), "n_divergent": int(ndiv),}

# Run MMCM Inversion with PyMC
def mcmc_inversion(
    yobs, ybg, ysem, srr, prior_emiss, prior_emiss_noncity, city_mask,
    draws=config_nDraws, tune=config_nTune, chains=config_nChains,
    cores=None, seed= 20240101,):

    settings = {
    "emiss_city_sigma":    1.0,    # LogNormal sigma on emiss_city
    "emiss_noncity_scale": 1e-3,   # emiss_noncity prior median = scale * prior_emiss
    "emiss_noncity_sigma": 0.5,    # LogNormal sigma on emiss_noncity
    "bg_sigma":            0.5,    # Normal sigma on background
    "repeatability_lower": 0.04,   # Uniform lower bound
    "repeatability_upper": 1.5,    # Uniform upper bound
    "model_error_sigma":   0.5,    # HalfNormal sigma on transport/model error          
    }

    if cores is None:
        cores = min(chains, os.cpu_count() or 1)

    ntime       = yobs.shape[0]
    city_srr    = np.array([np.nansum(srr[t] *  city_mask) for t in range(ntime)])
    noncity_srr = np.array([np.nansum(srr[t] * ~city_mask) for t in range(ntime)])

    valid = np.isfinite(yobs)
    if valid.sum() == 0:
        raise RuntimeError("All timesteps are NaN — nothing to invert.")

    obs_se = ysem.copy()
    fallback = np.nanmean(ysem[valid])
    obs_se[valid & ~np.isfinite(obs_se)] = fallback

    ybg_is_series = np.asarray(ybg).ndim > 0
    ybg_arr = np.broadcast_to(np.asarray(ybg, dtype=float), yobs.shape)

    tcoord = np.arange(ntime)[valid]

    with pm.Model(coords={"t": tcoord}):
        emiss_city = pm.LogNormal("emiss_city", mu=np.log(prior_emiss), sigma=settings["emiss_city_sigma"], dims="t",)
        emiss_noncity = pm.LogNormal("emiss_noncity", mu=np.log(prior_emiss_noncity), sigma=settings["emiss_noncity_sigma"], dims="t",)

        # Shared BG across timesteps
        if ybg_is_series:
            bg_offset = pm.Normal("bg_offset", mu=0.0, sigma=settings["bg_sigma"])
            bg = pm.Deterministic("bg", ybg_arr[valid] + bg_offset, dims="t")
        else:
            bg = pm.Normal("bg", mu=float(ybg), sigma=settings["bg_sigma"])

        repeatability = pm.Uniform("repeatability", lower=settings["repeatability_lower"], upper=settings["repeatability_upper"],)
        model_error   = pm.HalfNormal("model_error", sigma=settings["model_error_sigma"])

        ypred = bg + emiss_city * city_srr[valid] + emiss_noncity * noncity_srr[valid]
        yerr  = pm.math.sqrt(obs_se[valid] ** 2 + repeatability**2 + model_error**2)

        pm.Normal("Y_obs", mu=ypred, sigma=yerr, observed=yobs[valid], dims="t")

        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=0.9,
            return_inferencedata=True,
            random_seed=seed,
            progressbar=False,
        )

        pm.sample_posterior_predictive(
            idata,
            extend_inferencedata=True,
            random_seed=seed,
            progressbar=False,
        )

    diagnostics = check_convergence(idata)
    return idata, valid, city_srr, noncity_srr, obs_se, diagnostics

# Build Output Dataset
def build_output_dataset(idata, valid, times, yobs, ystd, srr,
    city_srr, noncity_srr, city_mask,
    pulse_index, ybg, prior_emiss, ypred_prior,):

    ntime  = len(times)
    post   = idata.posterior
    chains = post.sizes["chain"]
    draws  = post.sizes["draw"]
    nsamples = chains * draws

    ybg_arr = np.broadcast_to(np.asarray(ybg, dtype=float), (ntime,)).copy()

    # Select Variables to Save
    posterior_vars = ["emiss_city", "emiss_noncity", "bg", "repeatability", "model_error"]
    shared_vars    = ["bg", "repeatability", "model_error"]
    quantiles      = [0.025, 0.25, 0.5, 0.75, 0.975]

    # Posterior Calculations
    full_posterior = {}
    for var in posterior_vars:
        arr = np.full((ntime, chains, draws), np.nan)
        v = post[var].values                    
        if v.ndim == 2:
            arr[valid] = v[np.newaxis, ...]     
        else:
            arr[valid] = np.moveaxis(v, -1, 0)  
        full_posterior[var] = arr

    ypred_draws = np.full((ntime, nsamples), np.nan)
    ppc = idata.posterior_predictive["Y_obs"].values   # (chain, draw, t)
    ypred_draws[valid] = np.moveaxis(ppc, -1, 0).reshape(valid.sum(), nsamples)

    data_vars = {}
    for var in posterior_vars:
        data_vars[f"posterior_{var}"] = (("time", "chain", "draw"), full_posterior[var])

    for var in posterior_vars:
        flat = full_posterior[var].reshape(ntime, -1)
        data_vars[f"{var}_mean"] = (("time",), np.nanmean(flat, axis=-1))
        data_vars[f"{var}_std"]  = (("time",), np.nanstd(flat,  axis=-1))
        for q in quantiles:
            label = f"q{int(q * 1000):04d}"
            data_vars[f"{var}_{label}"] = (("time",), np.nanquantile(flat, q, axis=-1))

    data_vars["ypred_draws"] = (("time", "sample"), ypred_draws)
    data_vars["ypred_mean"]  = (("time",), np.nanmean(ypred_draws, axis=-1))
    data_vars["ypred_std"]   = (("time",), np.nanstd(ypred_draws,  axis=-1))
    for q in quantiles:
        label = f"q{int(q * 1000):04d}"
        data_vars[f"ypred_{label}"] = (("time",), np.nanquantile(ypred_draws, q, axis=-1))

    data_vars["yobs"]        = (("time",), yobs)
    data_vars["ystd"]        = (("time",), ystd)
    data_vars["ybg"]         = (("time",), ybg_arr)
    data_vars["ypred_prior"] = (("time",), ypred_prior)
    data_vars["pulse_index"] = (("time",), pulse_index.astype(int))
    data_vars["city_srr"]    = (("time",), city_srr)
    data_vars["noncity_srr"] = (("time",), noncity_srr)
    data_vars["prior_emiss"] = ((), prior_emiss)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time":   ("time",   times),
            "chain":  ("chain",  np.arange(chains)),
            "draw":   ("draw",   np.arange(draws)),
            "sample": ("sample", np.arange(nsamples)),
        },
    )

    for name in ds.data_vars:
        if name.startswith(("emiss_", "posterior_emiss_", "prior_emiss")):
            ds[name].attrs["units"] = "umol s-1" 
        elif name.startswith(("yobs", "ybg", "ypred", "posterior_bg", "bg_")):
            ds[name].attrs["units"] = "ppb" 
        elif name.startswith(("repeatability", "model_error",
                              "posterior_repeatability", "posterior_model_error")):
            ds[name].attrs["units"] = "ppb" 

    ds.attrs.update(
        {
            "description":     "Bayesian MCMC inversion of N2O emissions",
            "emission_units":  "umol s-1",
            "report_units":    "g hr-1",
            "conv_umol_s_to_g_hr": float(1e-6 * 44.013  * 3600),
            "model_structure": (
                "Single model over all valid timesteps. emiss_city and "
                "emiss_noncity are per-timestep and independent; bg, "
                "repeatability and model_error are shared across timesteps and "
                "are broadcast along the time axis on output."
            ),
            "ystd_definition": (
                "Standard error of the mean of the +/-15 min observation window "
                "(std/sqrt(n)); this is the quantity entering the likelihood."
            ),
            "ypred_definition": "Posterior predictive draws of Y_obs (includes observation and model error).",
            "prior_emiss":     float(prior_emiss),
            "ybg":             float(np.nanmean(ybg_arr)),
            "city_mask_cells": int(city_mask.sum()),
            "chains":          chains,
            "draws_per_chain": draws,
            "quantiles":       str(quantiles),
            "posterior_vars":  str(posterior_vars),
            "shared_vars":     str(shared_vars),
        }
    )
    return ds

# Run Inverion 
def run_inversion(
    year, obs_folder, backtraj_folder,
    draws, tune, chains, cores, seed=20240101,
    prior_norm="city",
    bg_method="median",
    prior_emiss_scale=1.0,
):

    inputs = preprocess_inputs(year, obs_folder, backtraj_folder)
    times, yobs, ystd, ysem, nobs, srr, lats, lons = inputs

    city_mask = define_city_mask(srr, lats, lons)

    ybg = estimate_background(times, yobs, method=bg_method)

    prior_emiss, prior_emiss_noncity, ypred_prior, pulse_index = get_prior_emissions(
        yobs, ystd, srr, city_mask, ybg,
        prior_norm=prior_norm,
        prior_emiss_scale=prior_emiss_scale,
    )

    idata, valid, city_srr, noncity_srr, obs_se, diagnostics = mcmc_inversion(
        yobs, ybg, ysem, srr, prior_emiss, prior_emiss_noncity, city_mask,
        draws=draws, tune=tune, chains=chains, cores=cores, seed=seed,
    )

    ds = build_output_dataset(
        idata=idata,
        valid=valid,
        times=times,
        yobs=yobs,
        ystd=obs_se,         
        srr=srr,
        city_srr=city_srr,
        noncity_srr=noncity_srr,
        city_mask=city_mask,
        pulse_index=pulse_index,
        ybg=ybg,
        prior_emiss=prior_emiss,
        ypred_prior=ypred_prior,
    )

    ds.attrs["prior_norm"]          = prior_norm
    ds.attrs["prior_emiss_noncity"] = float(prior_emiss_noncity)
    ds.attrs["prior_emiss_scale"]   = float(prior_emiss_scale)
    ds.attrs["bg_method"]           = bg_method
    ds.attrs["seed"]                = seed
    ds.attrs.update(diagnostics)

    return ds

# Main Execution
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ds = run_inversion(
        year=args.year,
        obs_folder=config_obs_path,
        backtraj_folder=config_stilt_path,
        draws=config_nDraws,
        tune=config_nTune,
        chains=config_nChains,
        cores=args.cores,
    )

    outfile = os.path.join(args.outdir, f"inversionResults_{args.year}.nc",)
    ds.to_netcdf(outfile)
    print(f"Saved: {outfile}")
    print(ds)