import os
import datetime
import logging

import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.io import fits
from astropy.table import Table

from xspec import *
Fit.query = "no"
import bxa.xspec as bxa

logger = logging.getLogger()


def _build_model_expression(model_name: str) -> str:
    if model_name == "powerlaw":
        return "constant*phabs*cflux*zpowerlw"
    if model_name == "apec_single":
        return "constant*phabs*cflux*apec"
    if model_name == "blackbody":
        return "constant*phabs*cflux*bbody"
    if model_name == "bremss":
        return "constant*phabs*cflux*bremss"
    raise ValueError(f"Unknown model: {model_name}")


def _set_group1_physical_parameters(model, model_name: str, redshift: float, flux_band=(0.5, 10.0)):
    # Inter-instrument normalization: reference instrument/group fixed to 1.
    model.constant.factor.values = "1.0"
    model.constant.factor.frozen = True

    model.phabs.nH.values = "0.05,,0.001,0.001,10.0,10.0"

    if model_name == "powerlaw":
        model.zpowerlw.Redshift = redshift
        model.zpowerlw.Redshift.frozen = True
        model.zpowerlw.PhoIndex.values = "2.0,,1.0,1.0,3.0,3.0"
        model.zpowerlw.norm.frozen = True
    elif model_name == "apec_single":
        model.apec.kT.values = "1.0,,0.1,0.1,10.0,10.0"
        model.apec.norm.frozen = True
    elif model_name == "blackbody":
        model.bbody.kT.values = "0.1,,0.01,0.01,2.0,2.0"
        model.bbody.norm.frozen = True
    elif model_name == "bremss":
        model.bremss.kT.values = "5.0,,0.1,0.1,20.0,20.0"
        model.bremss.norm.frozen = True

    model.cflux.Emin = flux_band[0]
    model.cflux.Emax = flux_band[1]
    model.cflux.lg10Flux.values = "-12.0,,-15.0,-15.0,-9.0,-9.0"


def _add_physical_priors(model, model_name: str):
    priors = [bxa.create_uniform_prior_for(model, model.phabs.nH)]

    if model_name == "powerlaw":
        priors.append(bxa.create_uniform_prior_for(model, model.zpowerlw.PhoIndex))
    elif model_name == "apec_single":
        priors.append(bxa.create_uniform_prior_for(model, model.apec.kT))
    elif model_name == "blackbody":
        priors.append(bxa.create_uniform_prior_for(model, model.bbody.kT))
    elif model_name == "bremss":
        priors.append(bxa.create_uniform_prior_for(model, model.bremss.kT))

    priors.append(bxa.create_uniform_prior_for(model, model.cflux.lg10Flux))
    return priors


def _load_spectra_in_separate_groups(spectrum_files, background_files, rmf_files, arf_files):
    if not (len(spectrum_files) == len(background_files) == len(rmf_files) == len(arf_files)):
        raise ValueError("Spectrum, background, RMF and ARF lists must have the same length")
    if len(spectrum_files) == 0:
        raise ValueError("No spectra provided")

    AllData.clear()
    expr = " ".join([f"{i+1}:{i+1} {spectrum_files[i]}" for i in range(len(spectrum_files))])
    AllData(expr)

    for i in range(len(spectrum_files)):
        s = AllData(i + 1)
        s.background = background_files[i]
        s.response = rmf_files[i]
        s.response.arf = arf_files[i]


def get_model_and_priors(model_name, redshift=0.0, flux_band=(0.5, 10.0), n_groups=1,
                         iin_bounds=(1.0e-3, 5.0)):
    """
    Build a simultaneous-fit model. Physical parameters are shared across groups,
    while inter-instrument normalization (IIN) is implemented as a multiplicative
    constant with group 1 fixed at 1 and groups 2..N free.
    """
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")

    model = Model(_build_model_expression(model_name))
    _set_group1_physical_parameters(model, model_name, redshift, flux_band=flux_band)

    priors = _add_physical_priors(model, model_name)

    low_iin, high_iin = iin_bounds
    if low_iin <= 0:
        raise ValueError("IIN lower bound must be > 0 if you want log10(IIN)")

    # Secondary groups are linked to group 1 by default in XSPEC.
    # We keep the physical parameters linked, and untie only the multiplicative constant.
    for group_index in range(2, n_groups + 1):
        gmod = AllModels(group_index)
        gmod.constant.factor.untie()
        gmod.constant.factor.frozen = False
        gmod.constant.factor.values = f"1.0,,{low_iin},{low_iin},{high_iin},{high_iin}"
        priors.append(bxa.create_uniform_prior_for(gmod, gmod.constant.factor))

    return model, priors


def check_background_fit(spectrum_file, background_file, rmf_file, arf_file,
                         model_name, redshift, logger, srcid="unknown"):
    """
    Returns a quick-fit p-value for the background gate, or None on failure.
    This operates on a single spectrum only.
    """
    try:
        AllData.clear()
        AllModels.clear()

        s = Spectrum(spectrum_file)
        s.background = background_file
        s.response = rmf_file
        s.response.arf = arf_file

        AllData.ignore("**-0.3 10.0-**")

        _model, _priors = get_model_and_priors(model_name, redshift, n_groups=1)
        Fit.perform()

        from scipy.stats import chi2
        bg_stat = Fit.testStatistic()
        dof = Fit.dof
        pval = 1.0 - chi2.cdf(bg_stat, dof)

        if (pval is None) or (not np.isfinite(pval)) or (pval < 0.01):
            logger.warning(f"   Background check: p={pval:.4f} (<0.01) -> NOT OK for {srcid}")
            return None

        logger.info(f"   Background check: p={pval:.4f} -> OK for {srcid}")
        return pval

    except Exception as e:
        logger.warning(f"   Background check failed for {srcid}: {e}")
        return None


def fit_spectrum_bxa(spectrum_files, background_files, rmf_files, arf_files,
                     redshift=0.0, model_name="powerlaw",
                     output_base="bxa_fit_results", srcid="unknown", log_file="fit_spectrum_bxa.log"):
    """
    Simultaneous BXA fit for 1..N spectra.

    The current implementation defines IIN only when N>1. Group 1 is the reference
    instrument with constant factor fixed to 1. The fitted constant(s) for groups 2..N
    are the IIN parameter(s).
    """
    logger.info("\n")
    logger.info(f"Starting BXA fit on {len(spectrum_files)} spectrum/s for SRCID {srcid}")

    try:
        workdir = os.path.dirname(os.path.abspath(spectrum_files[0]))
        os.chdir(workdir)
        logger.info(f"   Changing focus to {workdir}")
    except Exception:
        pass

    AllData.clear()
    AllModels.clear()
    Fit.statMethod = "cstat"
    Plot.device = "/null"

    _load_spectra_in_separate_groups(spectrum_files, background_files, rmf_files, arf_files)
    AllData.ignore("**-0.3 10.0-**")

    model, priors_list = get_model_and_priors(model_name, redshift, n_groups=len(spectrum_files))

    timestamp = datetime.datetime.now().strftime("%d%m%Y_%H%M")
    model_dirname = f"{model_name}_{timestamp}"
    output_dir = os.path.abspath(os.path.join(output_base, str(srcid), model_dirname))
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"   Setting the output directory for the fits {output_dir}")

    solver = bxa.BXASolver(transformations=priors_list,
                           outputfiles_basename=os.path.join(output_dir, ""))
    solver.run(resume=False)

    chain_file = os.path.join(output_dir, "chain.fits")
    if not os.path.exists(chain_file):
        logger.error(f"   Chain file {chain_file} not found after BXA run")
        return {"flag": 4}

    with fits.open(chain_file) as hdul:
        samples = hdul[1].data
        param_names = [n for n in samples.names if n != "FIT_STATISTIC"]
        samples_array = np.column_stack([samples[name] for name in param_names])

    stds = np.std(samples_array, axis=0)
    valid_cols = stds > 0
    filtered_samples = samples_array[:, valid_cols]
    filtered_labels = [label for i, label in enumerate(param_names) if valid_cols[i]]

    if filtered_samples.shape[1] > 0:
        fig = corner.corner(filtered_samples, labels=filtered_labels, show_titles=True, title_fmt=".3e")
        corner_path = os.path.join(output_dir, "corner.png")
        fig.savefig(corner_path)
        plt.close(fig)
        logger.info(f"   Saved corner plot to file {corner_path}")

    posterior_median = np.median(samples_array, axis=0)
    posterior_p16 = np.percentile(samples_array, 16, axis=0)
    posterior_p84 = np.percentile(samples_array, 84, axis=0)

    return {
        "parameter_names": param_names,
        "posterior_median": posterior_median,
        "posterior_p16": posterior_p16,
        "posterior_p84": posterior_p84,
        "output_dir": output_dir,
        "flag": 0,
    }


def _extract_chain_summaries(chain_file):
    with fits.open(chain_file) as hdul:
        samples = hdul[1].data
        param_names = [n for n in hdul[1].columns.names if n != "FIT_STATISTIC"]

        summaries = {}
        factor_cols = []
        for pname in param_names:
            vals = np.asarray(samples[pname], dtype=float)
            summaries[pname] = {
                "median": float(np.percentile(vals, 50)),
                "p16": float(np.percentile(vals, 16)),
                "p84": float(np.percentile(vals, 84)),
            }
            if pname.startswith("factor__"):
                factor_cols.append(pname)

        derived = {}
        if len(factor_cols) == 1:
            vals = np.asarray(samples[factor_cols[0]], dtype=float)
            good = np.isfinite(vals) & (vals > 0)
            if np.any(good):
                logvals = np.log10(vals[good])
                derived["IIN"] = {
                    "median": float(np.percentile(vals[good], 50)),
                    "p16": float(np.percentile(vals[good], 16)),
                    "p84": float(np.percentile(vals[good], 84)),
                }
                derived["log10IIN"] = {
                    "median": float(np.percentile(logvals, 50)),
                    "p16": float(np.percentile(logvals, 16)),
                    "p84": float(np.percentile(logvals, 84)),
                }
        elif len(factor_cols) > 1:
            for i, col in enumerate(sorted(factor_cols), start=2):
                vals = np.asarray(samples[col], dtype=float)
                good = np.isfinite(vals) & (vals > 0)
                if not np.any(good):
                    continue
                logvals = np.log10(vals[good])
                derived[f"IIN_g{i}"] = {
                    "median": float(np.percentile(vals[good], 50)),
                    "p16": float(np.percentile(vals[good], 16)),
                    "p84": float(np.percentile(vals[good], 84)),
                }
                derived[f"log10IIN_g{i}"] = {
                    "median": float(np.percentile(logvals, 50)),
                    "p16": float(np.percentile(logvals, 16)),
                    "p84": float(np.percentile(logvals, 84)),
                }

    return summaries, derived


def export_bxa_results_to_fits(srcid, output_base="bxa_fit_results", fits_filename="fit_results.fits",
                               log_file="fit_spectrum_bxa.log", global_results=False):
    src_dir = os.path.join(output_base, str(srcid))
    fits_path = os.path.join(output_base if global_results else src_dir, fits_filename)

    logger.info("\n")
    logger.info(f"Exporting BXA fit results to FITS file {fits_path}")

    os.makedirs(output_base, exist_ok=True)

    model_dirs = sorted([d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))])
    short_map = {"powerlaw": "PL", "blackbody": "BB", "bremss": "BR", "apec_single": "AP"}

    model_data = {}
    for mdir in model_dirs:
        for long_name, short_name in short_map.items():
            if mdir.startswith(long_name):
                model_chain = os.path.join(src_dir, mdir, "chain.fits")
                if os.path.exists(model_chain):
                    summaries, derived = _extract_chain_summaries(model_chain)
                    model_data[short_name] = {"params": summaries, "derived": derived}

    if os.path.exists(fits_path):
        table = Table.read(fits_path)
    else:
        table = Table({"SRCID": [srcid]})

    src_mask = table["SRCID"] == srcid
    if np.any(src_mask):
        idx = np.where(src_mask)[0][0]
    else:
        if "SRCID" not in table.colnames:
            table["SRCID"] = np.array([], dtype=int)
        table.add_row({"SRCID": srcid})
        idx = len(table) - 1

    for model_short, pdata in model_data.items():
        for pname, summary in pdata["params"].items():
            for suffix, val in summary.items():
                col = f"{model_short}_{pname}_{suffix}"
                if col not in table.colnames:
                    table[col] = np.full(len(table), np.nan)
                table[col][idx] = val

        for dname, summary in pdata["derived"].items():
            for suffix, val in summary.items():
                col = f"{model_short}_{dname}_{suffix}"
                if col not in table.colnames:
                    table[col] = np.full(len(table), np.nan)
                table[col][idx] = val

    table.write(fits_path, overwrite=True)
