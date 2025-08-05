from __future__ import division

import os
import pdb
from datetime import datetime
import sys

import astropy.io.fits as pyfits
from astroquery.mast import Catalogs
import matplotlib.pyplot as plt
import numpy as np

#import calistar
import csv

import matplotlib
matplotlib.rcParams.update({'font.size': 14})

# we need astroquery to get magnitudes, coordinates, etc.
from astroquery.simbad import Simbad
from astropy import units as u
from astropy.coordinates import SkyCoord

# to define abstract method
from abc import ABC, abstractmethod

# add some votable fields to get the magnitudes, proper motion, and plx required in acq template
Simbad.add_votable_fields('G')
Simbad.add_votable_fields('K')
Simbad.add_votable_fields('H')
Simbad.add_votable_fields('pmdec')
Simbad.add_votable_fields('pmra')
Simbad.add_votable_fields('plx_value')



# =============================================================================
# 
# =============================================================================

kmag_bins_med_s = [3.25, 5.50, 7.00, 8.25]  # vegamag, based on beta Pic
dit_bins_med_s = [0.3, 1.0, 3.0]  # s
ndit_bins_med_s = [96, 64, 16]
kmag_bins_high_s = [1.75, 3.25, 4.50, 5.75]  # vegamag
dit_bins_high_s = [0.3, 1.0, 3.0]  # s
ndit_bins_high_s = [96, 64, 16]


def compute_pointings(rho = 100, phi0 = 5., npts = 9, nsplit = 3):
    """
	rho = 100.  # mas
	phi0 = 5.  # deg, PA offset of first pointing
	npts = 9  # number of companion pointings along rho mas circle
	nsplit = 3  # split npts total pointings in nsplit separate OBs
    """
    # Compute companion pointings.
    phis = np.array([i * 360. / npts for i in range(npts)]) + phi0  # deg
    ras = rho * np.sin(np.deg2rad(phis))  # mas
    decs = rho * np.cos(np.deg2rad(phis))  # mas
    return phis, ras, decs

def query_simbad_gaiadr3(gaiaid):
    name = f'Gaia DR3 {gaiaid}'
    table = Simbad.query_object(name)
    if len(table) < 1:
        name = f'Gaia EDR3 {gaiaid}'
        table = Simbad.query_object(name)
    return table.to_pandas()
    
def query_simbad_object_name(name):
    table = Simbad.query_object(name)
    return table.to_pandas()

def make_ymlfiles(nsplit, npts, runid, gaiaid, phis, ras, decs, objname):
    #try:
    #    target_table = query_simbad_gaiadr3(gaiaid)
    #except Warning:
    if objname is not None:
        target_table = query_simbad_object_name(objname)
    elif objname == None:
        target_table = query_simbad_gaiadr3(gaiaid)
        objname = target_table.main_id.values[0]
    # Loop through number of splits.
    for split in range(nsplit):

        # Populate YML file.
        output = 'setup:\n'
        output += f'  run_id: {runid}\n'
        date = datetime.today().strftime('%Y-%m-%d')
        output += f'  date: {date}\n'
        output += f'  folder: {objname}\n'
        if 'K' in target_table.columns:
            kmag = target_table.K.values[0]
            if kmag >= kmag_bins_med_s[0]:
                mode = 'MED'
                kmag_bins_s = kmag_bins_med_s
                dit_bins_s = dit_bins_med_s
                ndit_bins_s = ndit_bins_med_s
                output += '  INS.SPEC.RES: "MED"\n'
            else:
                mode = 'HIGH'
                kmag_bins_s = kmag_bins_high_s
                dit_bins_s = dit_bins_high_s
                ndit_bins_s = ndit_bins_high_s
                output += '  INS.SPEC.RES: "HIGH"\n'
        else:
            raise UserWarning('No K-band magnitude found!')
        output += '  INS.SPEC.POL: "OUT"\n'
        output += '  ISS.BASELINE: ["UTs"]\n'
        output += '  ISS.VLTITYPE: ["astrometry"]\n'
        output += '  SEQ.MET.MODE: ON\n'
        output += '  concatenation: none\n'
        output += '  constraints:\n'
        output += '    skyTransparency: "Variable, thin cirrus"\n'
        output += '    airmass: 1.6\n'
        output += '    moonDistance: 10\n'
        output += '    atm: 85%\n'
        output += '\n'
        output += 'ObservingBlocks:\n'
        output += '  SCI_' + objname + '_search%.0f:\n' % (split + 1)
        output += '    description: Searching for new inner planet\n'
        output += '    mode: dual_on\n'
        output += '    target: ' + objname + '\n'
        if 'G' in target_table.columns:
            output += '    g_mag: %.3f\n' % target_table.G.values[0]
        if 'K' in target_table.columns:
            output += '    k_mag: %.3f\n' % target_table.K.values[0]
        if 'H' in target_table.columns:
            output += '    h_mag: %.3f\n' % target_table.H.values[0]
        if kmag < 5.50:
            ndit_c = 96
            dit_c = 3
        else:
            ndit_c = 32
            dit_c = 10
        output += '    objects:\n'
        output += '      s:\n'
        output += '        name: Central star\n'
        dit_s = dit_bins_s[np.digitize(kmag, kmag_bins_s) - 1]
        if dit_s < (1. - 1e-10):
            dit_s = '%.1f' % dit_s
        else:
            dit_s = '%.0f' % dit_s
        output += '        DET2.DIT: %s\n' % dit_s
        ndit_s = ndit_bins_s[np.digitize(kmag, kmag_bins_s) - 1]
        output += '        DET2.NDIT.OBJECT: %.0f\n' % ndit_s
        output += '        DET2.NDIT.SKY: %.0f\n' % ndit_s
        output += '        coord_syst: radec\n'
        output += '        coord: [0, 0]\n'
        for j in range(npts // nsplit):
            jj = split * npts // nsplit + j
            ind = split * npts // nsplit + j + 1
            output += '      c%.0f:\n' % ind
            output += '        name: Companion pointing %.0f\n' % ind
            output += '        DET2.DIT: %s\n' % dit_c
            output += '        DET2.NDIT.OBJECT: %.0f\n' % ndit_c
            output += '        DET2.NDIT.SKY: %.0f\n' % ndit_c
            output += '        coord_syst: radec\n'
            output += '        coord: [%.1f, %.1f]\n' % (ras[jj], decs[jj])
        output += '    sequence:\n'
        # output += '      - s sky\n'
        # for j in range(npts // nsplit):
        #     ind = split * npts // nsplit + j + 1
        #     if j % 2 == 1:
        #         output += '      - c%.0f c%.0f sky\n' % (ind, ind)
        #     else:
        #         output += '      - c%.0f c%.0f\n' % (ind, ind)
        #     if j % 2 == 1:
        #         output += '      - s sky\n'
        #     else:
        #         output += '      - s\n'
        if kmag < 5.50:
            output += '      - s sky\n'
            output += '      - c%.0f c%.0f sky c%.0f\n' % (3 * split + 1, 3 * split + 2, 3 * split + 3)
            output += '      - s\n'
            output += '      - c%.0f sky c%.0f c%.0f\n' % (3 * split + 1, 3 * split + 2, 3 * split + 3)
            output += '      - s sky\n'
        else:
            output += '      - s sky\n'
            output += '      - c%.0f c%.0f c%.0f sky\n' % (3 * split + 1, 3 * split + 2, 3 * split + 3)
            output += '      - s sky\n'
            output += '      - c%.0f c%.0f c%.0f\n' % (3 * split + 1, 3 * split + 2, 3 * split + 3)
            output += '      - s\n'
        output += '    calib: False\n'

        # Write YML file.
        ymlpath = odir + 'gaia_dr3_' + str(gaiaid) +\
            '_split%.0f.yml' % (split + 1)
        with open(ymlpath, 'w') as ymlfile:
            ymlfile.write(output)

        # Send OB to P2.
        os.system('python create_obs.py ' + ymlpath + ' --nogui')
        #os.system('python create_obs.py ' + ymlpath + ' --nogui')

def send_to_p2(gaiaids, runid, odir, rho = 100, phi0 = 5., npts = 9, nsplit = 3, objnames=None):
    phis, ra, decs = compute_pointings(rho,phi0,npts,nsplit)
    for i in range(len(gaiaids)):
        gaiaid = gaiaids[i]
        print(f'************* Gaia DR3 {gaiaid} ***************')
        if objnames is not None:
            objname = objnames[i]
        else:
            objname = None
        make_ymlfiles(nsplit, npts, runid, gaiaid, phis, ra, decs, objname)
        b = c
    
if __name__ == '__main__':    

    # list of targets
    path_targlist = '/Users/svach/gravity/gravity_gaia_target_lists/2025-08-05_P116_NewOBs.csv'
    #where the yml files will go
    odir = '/Users/svach/p2Gravity-fork/yml_outfiles/'
    #run id for P2 tool
    runid = '116.29AG.001'
    
    import pandas as pd
    df = pd.read_csv(path_targlist)
    df = df.drop_duplicates('GDR3')
    gaiaids = df.GDR3.values.astype(int)
    names = df.Name.values
    send_to_p2(gaiaids, runid, odir, objnames = names)


