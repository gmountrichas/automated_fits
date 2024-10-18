import os
import numpy as np
from astropy.io import fits
import logging


logger = logging.getLogger(__name__)



# Function to check which spectra are suitable for fitting
def check_spectra(list_spectra, responses_dir, output_dir, log_file):
    """
    Check the spectra for a given list of spectra. The good spectra are ungrouped and copied to output_dir
       together with soft links to their corresponding background, arf and rmf files
    
    If no pn spectra are selected, one element is added to pn_spectra with the highest flag that applies to all spectra
    Similarly for MOS
    
    
    
    Parameters:
    - list_spectra (str): List of spectra in data_dir
    - responses_dir (str): Top directory with the response matrices (rmf) for pn and MOS
    - output_dir (str): Directory to copy the good spectra, background, arf and rmf files to
    - log_file (str): The log file to write the messages.
    Returns:
    - pn_spectra (list): A list of tuples containing the full path and name for the good pn spectra, total source counts, total background counts, total net counts, exposure time, and a flag
    - mos_spectra (list): A list of tuples containing the full path and name for the good mos spectra, total source counts, total background counts, total net counts and a flag

    """

    pn_spectra = []
    mos_spectra = []
    pn_flag=-1
    mos_flag=-1
    
    
    # Making output_dir and responses_dir absolute paths, as needed by symbolic links
    responses_dir=os.path.abspath(responses_dir)
    output_dir=os.path.abspath(output_dir)
        
    
    for spectrum_file in list_spectra:
        message=f'\n\nWorking on file {spectrum_file}'
        logger.info(message)
        # getting absolute paths
        spectrum_file=os.path.abspath(spectrum_file)
        # Extracting just the spectrum name for output messages
        name=spectrum_file.split('/')[-1]
        banner=name
        #
        # Check if the source spectrum exists
        #    going for next spectrum if it does not
        if not os.path.exists(spectrum_file):
            message = f"      spectrum:{banner} - Missing source spectrum"
            logger.info(message)
            continue
        else:
            # source spectrum exists, what about the background spectrum?
            #
            # background filename corresponding to the source
            background_file=spectrum_file.replace("SRSPEC","BGSPEC")
            # Check if the background spectrum exists
            #     going for the next spectrum if it does not
            if not os.path.exists(background_file):
                message = f"      spectrum:{banner} - Missing background spectrum {background_file}"
                logger.info(message)
                continue
            else:
                # both source and background specra exist, finding out whether it is a pn or MOS spectrum
                pn= (banner.find("PN")>0)
                # Check the counts in the background spectrum
                with fits.open(background_file) as bg_hdul:
                    bg_counts = bg_hdul[1].data['COUNTS'].sum()
                    bg_backscal=bg_hdul[1].header['BACKSCAL']
                
                if bg_counts > 0:  # Only proceed if background has counts
                    
                    with fits.open(spectrum_file) as sp_hdul:
                        sp_counts = sp_hdul[1].data['COUNTS'].sum()
                        sp_header=sp_hdul[1].header
                    
                    if sp_counts > 0:  # Only proceed if source spectrum has counts
                    
                        # verifying now that the arf file exists
                        arf_file=spectrum_file.replace('SRSPEC','SRCARF')
                        if not os.path.exists(arf_file):
                            message = f"      spectrum:{banner} - Missing arf file {arf_file}"
                            logger.info(message)
                            continue
                        else:
                            # verifying now that the rmf file exists
                            response=sp_header['RESPFILE']
                            # full path to and name of rmf file depend on whetherpn/MOS
                            if (pn):
                                # adding _v19.0 at the end of the root
                                response19=response.split('.')[0]+'_v19.0.rmf'
                                rmf_file=os.path.join(responses_dir+'/PN',response19)
                            else:
                                # finding out the resolution of the rmf
                                specdelt=sp_header['SPECDELT']
                                rmf_file=os.path.join(responses_dir+'/MOS','{:d}eV/'.format(specdelt)+response)
                            #
                            
                            if not os.path.exists(rmf_file):
                                message = f"      spectrum:{banner} - Missing rmf file {rmf_file}"
                                logger.info(message)
                                continue
                            else:
                                # all necessary files present
                                # calculating net counts
                                sp_backscal=sp_header['BACKSCAL']
                                sp_exp=sp_header['EXPOSURE']
                                sp_netcts=sp_counts-bg_counts*sp_backscal/bg_backscal
                                
                                if (sp_netcts >0):
                                    # This spectrum is suitable for fitting
                                    message = f"      spectrum:{banner} - Good for fitting"
                                    logger.info(message)
                                    #
                                    spectrum_fit=spectrum_file.split('/')[-1]
                                    spectrum_fit=os.path.join(output_dir,spectrum_fit)
                                    #
                                    # appending spectrum to be fitted to output list
                                    out_tuple=(spectrum_fit,sp_counts,bg_counts,sp_netcts,sp_exp,0)
                                    message=f"       (fit file, source counts, background counts, net counts, exposure time, flag) = {out_tuple} "
                                    logger.info(message)
                                    if (pn):
                                        pn_spectra.append(out_tuple)
                                    else:
                                        mos_spectra.append(out_tuple)
                                    #
                                    # linking spectrum, background and arf in output dir
                                    for file in [spectrum_file, background_file, arf_file]:
                                        fitfile=file.split('/')[-1]
                                        fitfile=os.path.join(output_dir,fitfile)
                                        if(os.path.islink(fitfile)):os.unlink(fitfile)
                                        os.symlink(file,fitfile)
                                    # 
                                    # rmf file has to be treated differently because
                                    #     its name for pn is harwired in header and is named differently
                                    rmf_fit=os.path.join(output_dir,response)
                                    if(os.path.islink(rmf_fit)):os.unlink(rmf_fit)
                                    os.symlink(rmf_file,rmf_fit)
                                else:
                                    message = f"      spectrum:{banner} - Source spectrum has <=0 net counts"
                                    logger.info(message)
                                    if (pn):
                                        pn_flag=max(pn_flag,2)
                                    else:
                                        mos_flag=max(mos_flag,2)
                                    #spectrum
                                #
                            #
                        #
                    else:
                        message = f"      spectrum:{banner} - Source spectrum has <=0 total counts"
                        logger.info(message)
                        if (pn):
                            pn_flag=max(pn_flag,2)
                        else:
                            mos_flag=max(mos_flag,2)
                        #
                else:
                    # no background counts for this spectrum: updating the corresponding maximum
                    #    flag and going for the next spectrum
                    message = f"      spectrum:{banner} - Background spectrum has <=0 total counts"
                    logger.info(message)
                    if (pn):
                        pn_flag=max(pn_flag,1)
                    else:
                        mos_flag=max(mos_flag,1)
                    #
                    continue
                #
            #
        #
        
    # if no useful pn spectra, formatting output
    if (len(pn_spectra)==0):
        pn_spectra.append(('none',np.nan,np.nan,np.nan,np.nan,pn_flag))

    # if no useful pn spectra, formatting output
    if (len(mos_spectra)==0):
        mos_spectra.append(('none',np.nan,np.nan,np.nan,np.nan,mos_flag))
            
    
    
    return pn_spectra,mos_spectra


def test_check_spectra():
    output_dir='./test_data/test'
    if not os.path.exists(output_dir) : os.mkdir(output_dir)
    # Set up logging
    log_file = os.path.join(output_dir, 'test_process_log.txt')
    logging.basicConfig(filename=log_file, level=logging.INFO)
    # common
    responses_dir='./test_data/RESPONSES'
    #
    # only pn
    list_spectra=['./test_data/0760940101/pps/P0760940101PNS003SRSPEC0017.FTZ']
    pn_list,mos_list=check_spectra(list_spectra, responses_dir, output_dir, log_file)
    # one pn spectrum
    assert len(pn_list)==1
    # that spectrum is suitable for fitting
    assert pn_list[0][5]==0
    # no MOS spectra suitable for fitting
    assert mos_list[0][5]==-1
    #
    # only MOS
    list_spectra=['./test_data/0760940101/pps/P0760940101M1S001SRSPEC0017.FTZ','./test_data/0760940101/pps/P0760940101M2S002SRSPEC0017.FTZ']
    pn_list,mos_list=check_spectra(list_spectra, responses_dir, output_dir, log_file)
    # no pn spectrum is suitable for fitting
    assert pn_list[0][5]==-1
    # two MOS spectra
    assert len(mos_list)==2
    # both MOS spectra suitable for fitting
    assert mos_list[0][5]==0
    assert mos_list[1][5]==0
    #
    # all spectra
    list_spectra=['./test_data/0760940101/pps/P0760940101PNS003SRSPEC0017.FTZ','./test_data/0760940101/pps/P0760940101M1S001SRSPEC0017.FTZ','./test_data/0760940101/pps/P0760940101M2S002SRSPEC0017.FTZ']
    pn_list,mos_list=check_spectra(list_spectra, responses_dir, output_dir, log_file)
    # one pn spectrum
    assert len(pn_list)==1
    # that spectrum is suitable for fitting
    assert pn_list[0][5]==0
    # two MOS spectra
    assert len(mos_list)==2
    # both MOS spectra suitable for fitting
    assert mos_list[0][5]==0
    assert mos_list[1][5]==0
  
    
