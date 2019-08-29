import sys
import os
import numpy as np
import glob
import astropy.table
from   astropy.table import Table, Column
from   astropy.io    import fits
import fitsio

import matplotlib.pyplot as pl

# import apcsv.util_newdefs as util

from desitarget.targetmask import desi_mask, bgs_mask, mws_mask

import multiprocessing as _mp
default_mp_proc = 4

############################################################
class FiberAssignFiles():
   
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self,path,survey='SV1'):
        """
        """
        self.tile_files   = glob_in_path(path,'tile-*.fits')
        self.headers      = self.load_fiber_headers()
       
        self.survey             = survey
        self.MWS_TARGET_NAME    = 'MWS_TARGET'
        self.BGS_TARGET_NAME    = 'BGS_TARGET'
        self.DESI_TARGET_NAME   = 'DESI_TARGET'

        if self.survey.startswith('SV'):
            for _ in [self.MWS_TARGET_NAME,self.BGS_TARGET_NAME,self.DESI_TARGET_NAME]:
                _ = '{}_{}'.format(self.survey,_)
                
            
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_fiber_data(self,ext='FIBERASSIGN',hdr=False,simple=False):
        """
        simple: Don't end up with dict['FIBERASSIGN'] = results.
        """
        data = dict()

        if isinstance(ext,list):
            extlist = ext
        else:
            extlist = [ext]

        for e in extlist:
            data[e] = dict()

        if hdr:
            data['_HEADER'] = dict()

        fiber_files = self.tile_files
        
        print('Have {} tiles'.format(len(fiber_files)))
        for fiber_file in fiber_files:
            filename    = os.path.basename(fiber_file)
            itile       = int(os.path.splitext(filename)[0].split('-')[1])
            for e in extlist:
                data[e][itile] = fitsio.read(fiber_file,e)
            if hdr:
                data['_HEADER'][itile] = fitsio.read_header(fiber_file)
                
        if simple:
            if len(extlist) == 1 and not hdr:
                data = data[e]
        return data  
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def load_fiber_headers(self):
        """
        """
        data = dict()

        fiber_files = self.tile_files
        print('Have {} tiles'.format(len(fiber_files)))
        for fiber_file in fiber_files:
            filename = os.path.basename(fiber_file)
            itile    = int(os.path.splitext(filename)[0].split('-')[1])
            h        = fitsio.read_header(fiber_file)
            data[itile] = h
        return data

############################################################
def glob_in_path(path,pattern):
    """
    """
    return glob.glob(os.path.join(path,pattern))

############################################################
def load_fiber_data_lean(path,ext='FIBERASSIGN',hdr=False):
    """
    """
    data = dict()
    
    if isinstance(ext,list):
        extlist = ext
    else:
        extlist = [ext]
    
    for e in extlist:
        data[e] = dict()
    
    if hdr:
        data['_HEADER'] = dict()
    
    fiber_files = glob_fibers(path)
    print('Have {} tiles'.format(len(fiber_files)))
    for fiber_file in fiber_files:
        filename    = os.path.basename(fiber_file)
        itile       = int(os.path.splitext(filename)[0].split('-')[1])
        for e in extlist:
            data[e][itile] = fitsio.read(fiber_file,e)
        if hdr:
            data['_HEADER'][itile] = fitsio.read_header(fiber_file)
    return data

############################################################
def load_fiber_data(path,ext='FIBERASSIGN',only_first=None):
    """
    """
    import gc
    data = dict()
    fiber_files = glob_fibers(path)
    print('Have {} tiles'.format(len(fiber_files)))
    
    if only_first is not None:
        fiber_files = fiber_files[:only_first]
        
    for fiber_file in fiber_files:
        filename    = os.path.basename(fiber_file)
        itile       = int(os.path.splitext(filename)[0].split('-')[1])
        f           = fits.open(fiber_file,memmap=False)
        data[itile] = f[ext].data
        f.close()
    return data


############################################################
def tile_dict_to_table(tile_dict):
    """
    tile_dict is a dict keyed by tile ID with FITSrec values.
    """
    t = list()
    l = list()
    for k,v in tile_dict.items():
        t.append(v)
        l.append(np.repeat(k,len(v)))
        
    t = Table(np.concatenate(t))
    t.add_column(Column(np.concatenate(l),'ITILE'))
    return t


############################################################
def format_cats(cats,ncols=60):
    """
    """
    fmt_string = '{{:{ncols:d}}}'.format(ncols=ncols)
    cats    = ' & '.join(cats)
    cat_str = list()
    while len(cats) > ncols:
        rindex = cats[0:ncols].rindex(' & ')+1
        cat_str.append(fmt_string.format(cats[0:ncols][0:rindex]))
        cats = cats[rindex:]
    cat_str.append(fmt_string.format(cats))
    
    return cat_str
    
############################################################
def report_fibers_mws(tiledata,ncols=60):
    """
    """
    fmt_string   = '{{:{ncols:d}}} {{:<10d}}'.format(ncols=ncols)
    non_mws_line = None
    
    lines = list()
    ubits, ubits_count = np.unique(tiledata['MWS_TARGET'],return_counts=True)
    for ubit, ubit_count in zip(ubits, ubits_count):
        cats = list()
        for _ in mws_mask.names(ubit):
            _ = _.replace('MWS_','')
            cats.append(_)

        cat_str = format_cats(cats)
        if len(cat_str) > 1:
            for _s in cat_str[:-1]:
                lines.append(_s)
                
        cat_str = cat_str[-1]
        blank   = len(cat_str.strip()) == 0
        
        if blank: cat_str = '(non-MWS targets)'
            
        line = fmt_string.format(cat_str,ubit_count)
        
        if blank:
            non_mws_line = line
        else:
            lines.append(line)
    
    for line in lines:
        print(line)
    
    if non_mws_line is not None:
        print()
        print(non_mws_line)

    return

############################################################
def plot_pie_target_classes(tiledata):
    """
    """
    ubits, ubits_count = np.unique(tiledata['MWS_TARGET'],return_counts=True)
    
    labels = [ubits_labels[u] if u in util.mws_bits_to_name else u for u in ubits]
    colors = [ubits_colors[u] if u in util.mws_bits_to_colors else 'None' for u in ubits]

    kwargs = dict()
    kwargs['labels'] = labels
    kwargs['colors'] = colors

    pl.pie(ubits_count,**kwargs)
    pl.axis('equal')
    
    return

############################################################
def survey_overlaps(tiledata):
    """
    Returns True for fibers that have MWS_ANY bit set in 
    DESI_TARGET and other DESI_TARGET bits set.
    """
    is_assigned = fibers_assigned(tiledata)
    
    # Define as having MWS_ANY and any bits other than MWS_ANY
    is_mws   = (tiledata['DESI_TARGET'] &  desi_mask.mask('MWS_ANY'))!= 0
    is_other = (tiledata['DESI_TARGET'] & ~desi_mask.mask('MWS_ANY'))!= 0
    return (is_assigned & is_mws & is_other)

############################################################
def report_survey_overlaps(tiledata):
    """
    """
    is_overlap = survey_overlaps(tiledata)
    print('{:d} fibers with both MWS and non-MWS target bits:'.format(is_overlap.sum()))
    
    udesi,udesi_count = np.unique(tiledata['DESI_TARGET'][is_overlap],
                                       return_counts=True)
    
    overlap_lines = list()
    for (_desi,_c) in zip(udesi,udesi_count):
        names = desi_mask.names(_desi)
        names.remove('MWS_ANY')
        desi_str = ' & '.join(names)
        
        _ = tiledata['DESI_TARGET'][is_overlap] == _desi
        mws_bits = np.unique(tiledata['MWS_TARGET'][is_overlap][_])
        
        mws_str = list()
        for _ in mws_bits:
            if _ in util.mws_bits_to_name:
                mws_str.append(util.mws_bits_to_name[_])
        mws_str = ' | '.join(mws_str)
        
        line = ' {:30s} {:4d} {:s}'.format(desi_str,_c,mws_str)
        overlap_lines.append(line)
        
    for line in (sorted(overlap_lines)):
        print(line)
        
    return