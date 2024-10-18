import argparse
import os
import sys
#import glob
#import numpy as np
#from astropy.io import fits
import bxa.xspec as bxa
import xspec
import logging

from read_stacked_catalog import read_stacked_catalog
from list_spectra import list_spectra
from check_spectra import check_spectra


logger = logging.getLogger(__name__)




# Function to perform the BXA fitting for a given SRCID and spectrum
def fit_with_bxa(srcid, obsid, spectrum_file, background_file, model_name, redshift, use_galabs, use_tbabs_table, output_dir, log_file):
    """
    Fits a spectrum using the BXA (Bayesian X-ray Analysis) method.

    Args:
        srcid (int): The source ID.
        obsid (int): The observation ID.
        spectrum_file (str): The path to the spectrum file.
        background_file (str): The path to the background file.
        model_name (str): The name of the model.
        redshift (float): The redshift value.
        use_galabs (bool): Whether to use galactic absorption.
        use_tbabs_table (bool): Whether to use the tbabs table.
        output_dir (str): The output directory.
        log_file (str): The path to the log file.

    Raises:
        Exception: If any error occurs during the fitting process.

    Returns:
        None
    """
    
    try:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load the spectrum and background data into XSPEC
        xspec.AllData.clear()
        xspec.AllData(spectrum_file)
        spectrum = xspec.AllData(1)
        spectrum.background = background_file

        # Define the model based on the specified model name
        model = xspec_model(model_name, redshift)

        # Apply additional model options
        if use_galabs:
            model.setPars(galabs=True)
        if use_tbabs_table:
            model.setPars(tbabs=True)

        # Configure and run the BXA fitting process
        fit = bxa.Fit(model, output_dir)
        fit.run()

        # Save the fitting results
        fit.results(output_dir + "/fit_results.fits")
    except Exception as e:
        # Log any errors encountered during the fitting process
        message = f"SRCID {srcid}: BXA fitting failed for OBSID {obsid} with model {model_name}. Error: {str(e)}"
        logger.error(message)

# Function to define the XSPEC model based on the model name
def xspec_model(model_name, redshift):
    """
    Create an XSPEC model based on the given model name and redshift.

    Parameters:
    - model_name (str): The name of the XSPEC model to create.
    - redshift (float): The redshift value to use for the model.

    Returns:
    - model (xspec.Model): The created XSPEC model.

    Raises:
    - ValueError: If the given model name is unknown.
    """
    if model_name == "powerlaw":
        model = xspec.Model("powerlaw")
    elif model_name == "blackbody":
        model = xspec.Model("bbody")
    elif model_name == "apec_single":
        model = xspec.Model("apec")
    elif model_name == "apec_apec":
        model = xspec.Model("apec + apec")
    elif model_name == "apec_apec_const":
        model = xspec.Model("apec + apec + constant")
    elif model_name == "bremss":
        model = xspec.Model("bremss")
    elif model_name == "powerlaw_blackbody":
        model = xspec.Model("powerlaw + bbody")
    elif model_name == "powerlaw_blackbody_const":
        model = xspec.Model("powerlaw + bbody + constant")
    elif model_name == "zpowlaw":
        model = xspec.Model(f"zpowerlw,zpowerlw.redshift={redshift}")
    elif model_name == "double_zpowlaw":
        model = xspec.Model(f"zpowerlw + zpowerlw,zpowerlw.redshift={redshift}")
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    return model

# Function to select the best spectrum based on SNR and perform fitting
def fit_spectrum(srcid, spectra, args, model_name, log_file):
    """
    Fits a spectrum using the BXA method.
    Parameters:
        srcid (str): The source ID.
        spectra (list): A list of spectra.
        args (Namespace): The command line arguments.
        model_name (str): The name of the model.
        log_file (str): The path to the log file.
    Returns:
        None
    """
    best_spectrum = max(spectra, key=lambda x: x[3])  # Choosing based on highest SNR
    obsid = best_spectrum[0]
    spectrum_file = f"{args.data_dir}/{srcid}/{obsid}_SRSPEC0001.FTZ"
    background_file = f"{args.data_dir}/{srcid}/{obsid}_BGSPEC0001.FTZ"
    
    # Log the fitting process and perform the BXA fitting
    message = f"Using BXA to fit {model_name} model for SRCID {srcid} OBSID {obsid}"
    logger.info(message)
    output_dir = os.path.join(args.data_dir, srcid)
    fit_with_bxa(srcid, spectrum_file, background_file, model_name, args.redshift, args.use_galabs, args.use_tbabs_table, output_dir, log_file)

# Main function to handle argument parsing and control the workflow
def main():
    parser = argparse.ArgumentParser()

    # Paths to data and scripts
    parser.add_argument("srcid", type=int, help="SRCID of the source whose spectra are to be fitted")
    parser.add_argument("data_dir", help="Path to the directory containing the data")
    parser.add_argument("script_dir", help="Path to the directory containing the scripts")
    parser.add_argument("responses_dir", help="Path to the directory containing the response matrices")
    parser.add_argument("output_dir", help="Path to the output directory")

    parser.add_argument("catalog", help="Stacked catalog FITS filename (including path)")
    parser.add_argument("output", help="Name to the output file with fit results")
    parser.add_argument("--init", action="store_true", help="initialize the directory")
    parser.add_argument("--combine", action="store_true", help="re-merge the spectra")
    parser.add_argument("--fit_bkg", action="store_true", help="fit the background model")
    parser.add_argument("--get_bkg_stat", action="store_true", help="get the bkg statistics")

    # Model fitting options
    parser.add_argument("--fit_pl", action="store_true", help="fit powerlaw")
    parser.add_argument("--fit_bb", action="store_true", help="fit blackbody")
    parser.add_argument("--fit_apec_single", action="store_true", help="fit apec_singe")
    parser.add_argument("--fit_apec_apec", action="store_true", help="fit apec_apec")
    parser.add_argument("--fit_apec_apec_const", action="store_true", help="fit apec_apec_const")
    parser.add_argument("--fit_bremss", action="store_true", help="fit bremss")
    parser.add_argument("--fit_bbpl", action="store_true", help="fit blackbody powerlaw")
    parser.add_argument("--fit_bbpl_const", action="store_true", help="fit blackbody powerlaw with constant normalization")
    parser.add_argument("--fit_bbpl_const2", action="store_true", help="fit blackbody powerlaw with constant normalization")
    parser.add_argument("--fit_zpl", action="store_true", help="fit redshifted powerlaw")
    parser.add_argument("--fit_zplpl", action="store_true", help="fit redshifted double powerlaw")

    # Additional options for model fitting
    parser.add_argument("--suffix", default=None, help="directory suffix to add to srcid on dataAGN. If left out, it is found automatically based on the best-fit parameters. Use --suffix='' to override this behavior.")
    parser.add_argument("--suffix2", default=None, help="directory suffix to add for det_there != det_use")
    parser.add_argument("--redshift", type=float, help="redshift to be used for redshifted models")
    parser.add_argument("--modelname", help="use a custom modelname i.e. src0001_{modelname}")
    parser.add_argument("--use_galabs", type=int, default=0, help="fix foreground galactic absorption for extragalactic sources. The galactic absorption is estimated automatically according to the source coordinates.")
    parser.add_argument("--use_tbabs_table", type=int, default=0, help="Use tbabs table instead of xspec tbabs")
    parser.add_argument("--spinfo", type=int, default=0, help="calculate spinfo")
    parser.add_argument("--use_xmmsas_version_new", action="store_true", help="use the new (wrong) xmmsas version")
    parser.add_argument("--overwrite", type=int, default=1, help="overwrite existing model")
    args = parser.parse_args()

    #
    srcid=int(args.srcid)

    # Define output directory and log file for the SRCID
    output_dir = os.path.join(args.output_dir, args.srcid)
    # Set up logging
    log_file = os.path.join(output_dir, f"{srcid}_process_log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO)

    
    #
    message=f'\n\n Working on SRCID {srcid} '
    logger.info(message)
    # Read the catalog and map each SRCID to its corresponding OBSIDs
    srcid_obsid_mapping = read_stacked_catalog(args.catalog,srcid)
    if(len(srcid_obsid_mapping)>0):
        # SRCID found in file
        nout=len(srcid_obsid_mapping[srcid])
        message=f"   SRCID {srcid} : {nout} OBS_ID,SRC_NUM tuples found "
        logger.info(message)
        message=f'      tuples {srcid_obsid_mapping}'
        logger.info(message)     
    else:
        # SRCID not found
        message = f"SRCID {srcid} not found in file {args.catalog}"
        logger.error(message)
        #
        # terminating the program with error 1 (srcid not found in catalogue file)
        sys.exit(1)
    #

    
    # Getting the list of spectra for SRCID actually present in the data directory
    srcid_list_spectra=list_spectra(srcid,srcid_obsid_mapping,args.data_dir)
    nspec=len(srcid_list_spectra)
    message=f'   {nspec} spectra found for SRCID {srcid}'
    logger.info(message)
    for spec in srcid_list_spectra:
        message=f'      Spectrum {spec}'
        logger.info(message)
    #
    if (nspec==0):
        message=f' No spectra found for SRCID {srcid}'
        logger.error(message)
        #
        # terminating the program with error 2 (no spectra found for srcid)
        sys.exit(2)
    #


    # Checking if the spectra present are suitable for fitting
    pn_list,mos_list=check_spectra(srcid_list_spectra, args.responses_dir, output_dir, log_file)
    #
    # number of pn spectra suitable for fitting, or with source or background counts
    npn=len(pn_list)
    #
    # number of MOS spectra suitable for fitting, or with source or background counts
    nmos=len(mos_list)
    #
    
    if (npn+nmos==0):
        message=f'No spectra suitable for fitting found for SRCID {srcid}'
        logger.error(message)
        #
        # terminating the program with error 3 (no spectra suitable for fitting for srcid)
        sys.exit(3)
    #
        

    # Bits below remain from original version, commenting them out
    # They will be superseded (eventually) by calls to merge_spectra,perform_spectrum_fitting...
    
    
    '''
    results = []

    for srcid, obsids in srcid_obsid_mapping.items():


        # Initialize log file for this SRCID
        message = f"Processing SRCID {srcid}"
        logger.info(message)

        # Check spectra and identify those that are suitable for fitting
        good_spectra = check_spectra(args.data_dir, srcid, obsids, log_file)
        if not good_spectra:
            message = f"SRCID {srcid}: No good spectra found for fitting"
            logger.info(message)
            continue

        # Perform fitting for each model type specified
        perform_spectrum_fitting(args, srcid, log_file, good_spectra)

        # Append results to the output list
        results.append((srcid, good_spectra))
    
    # Write the final results to the specified output file
    with open(args.output, 'w') as f:
        for result in results:
            srcid, spectra = result
            f.write(f"SRCID: {srcid}\n")
            for spectrum in spectra:
                f.write(f"  OBSID: {spectrum[0]}, Source Counts: {spectrum[1]}, Background Counts: {spectrum[2]}, SNR: {spectrum[3]:.2f}\n")

'''

# Function to perform the BXA fitting for a given SRCID based on selected model
def perform_spectrum_fitting(args, srcid, log_file, good_spectra):
    """
    Perform spectrum fitting for the given source ID and good spectra.

    Parameters:
    - args: The command line arguments.
    - srcid: The source ID.
    - log_file: The log file to write the fitting results.
    - good_spectra: The list of good spectra.

    Returns:
    None
    """
    if args.fit_pl:
        fit_spectrum(srcid, good_spectra, args, "powerlaw", log_file)
    if args.fit_bb:
        fit_spectrum(srcid, good_spectra, args, "blackbody", log_file)
    if args.fit_apec_single:
        fit_spectrum(srcid, good_spectra, args, "apec_single", log_file)
    if args.fit_apec_apec:
        fit_spectrum(srcid, good_spectra, args, "apec_apec", log_file)
    if args.fit_apec_apec_const:
        fit_spectrum(srcid, good_spectra, args, "apec_apec_const", log_file)
    if args.fit_bremss:
        fit_spectrum(srcid, good_spectra, args, "bremss", log_file)
    if args.fit_bbpl:
        fit_spectrum(srcid, good_spectra, args, "powerlaw_blackbody", log_file)
    if args.fit_bbpl_const:
        fit_spectrum(srcid, good_spectra, args, "powerlaw_blackbody_const", log_file)
    if args.fit_bbpl_const2:
        fit_spectrum(srcid, good_spectra, args, "powerlaw_blackbody_const2", log_file)
    if args.fit_zpl:
        fit_spectrum(srcid, good_spectra, args, "zpowlaw", log_file)
    if args.fit_zplpl:
        fit_spectrum(srcid, good_spectra, args, "double_zpowlaw", log_file)

if __name__ == "__main__":
    main()
