from astropy.io import fits


# Function to read the stacked 4XMM-DR11 catalog and map SRCID to corresponding OBSIDs
def read_stacked_catalog(catalog_file,srcid_ref):
    """
    Read a stacked catalog file and a SRCID and create a dictionary mapping each SRCID to its associated list of OBS_ID and SRC_NUM .

    Parameters:
    catalog_file (str): The path to the stacked catalog file.

    srcid_ref (long): the SRCID to be fitted

    Returns:
    dict: A dictionary associating the SRCID to its list of OBS_ID and SRC_NUM.
    """

    with fits.open(catalog_file) as hdul:
        catalog_data = hdul[1].data

    # Create a dictionary to map each SRCID to its list of OBS_ID/SRC_NUM
    srcid_obsid_mapping = {}
    # Flag to see if we have reached the SRCID yet
    found=False
    # loop over the rows in the input file
    for i in range(len(catalog_data)):
        srcid = catalog_data['SRCID'][i]
        obsid = catalog_data['OBS_ID'][i]
        srcnum=catalog_data['SRC_NUM'][i]

        # checking if this row corresponds to the input SRCID
        if srcid==srcid_ref:    
            if srcid in srcid_obsid_mapping:
                # second and consecutive rows appended
                srcid_obsid_mapping[srcid].append((obsid,srcnum))
            else:
                # ignoring the first row for each SRCID, because no OBS_ID on it
                # initializing the list of tuples (OBS_ID,SRC_NUM)
                srcid_obsid_mapping[srcid] = []
                # setting the flag
                found=True
        elif found:
            # all rows for the same SRCID are consecutive so, once SRCID has been found
            #     all following rows with different SRCID can be safely skipped
            break

    return srcid_obsid_mapping


def test_read_stacked_catalog():
      # test catalogue
      infile='./test_data/test_catalogue.fits'
      
      # srcid to check
      srcids=[]
      # this should not be in the file
      srcids.append(1000000000000000)
      # this should return 0 hits
      srcids.append(3072415020100239)
      # this should return 1 hit
      srcids.append(3040339010100035)
      # this should reutnr 5 hits
      srcids.append(3030408050100122)
      #
      # expected results
      results=[-1,0,1,5]
      
      for i in range(len(srcids)):
          srcid=srcids[i]
          result=results[i]
          dic=read_stacked_catalog(infile,srcid)
          if (len(dic)==0):
              # if SRCID not found, setting the value to -1
              ndic=-1
          else:
              ndic=len(dic[srcid])
          
          assert ndic==result


          
