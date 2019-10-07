import numpy as np
from pandeia.engine.calc_utils import build_default_calc
from pandeia.engine.perform_calculation import perform_calculation
import pandeia.engine


# This script was adapted from a short script written by Michael Maseda
# demonstrating how to set up emission line S/N calculations in pandeia.

filterWlDict = {'clear':{'low':0.7,'high':5.1},
                'f070lp':{'low':0.7,'high':1.2},
                'f100lp':{'low':1.1,'high':1.8},
                'f170lp':{'low':1.7,'high':3.1},
                'f290lp':{'low':3.0,'high':5.1}}



def bn(n):
    """For sersic calculations
    """
    b = (2. * n) - 1./3. + 4. / (405. * n) + 46. / (25515. * n**2.)
    b += 131. / (1148175. * n**3.) - 2194697. / (30690717750. * n**4.)
    return b


def majorminor(n, re, ellip):
    #ellipticity is defined as (major-minor)/major
    scale_length = re / (bn(n)**n)
    #scale length is the circularized radius, i.e. r_scale = sqrt(a*b)
    major_axis = scale_length / np.sqrt(1. - ellip)
    minor_axis = scale_length * np.sqrt(1. - ellip)
    return (major_axis, minor_axis)


def sn_user_spec(inputs, disperser='prism', filt='clear',
                 ngroup=19, nint=2, nexp=36):
    wl = inputs['wl']  # in microns
    spec = inputs['spec']  # in mJy
    xoff = inputs['xoff']
    yoff = inputs['yoff']

    # --- Build a configuration ---
    configuration = build_default_calc('jwst', 'nirspec', 'msa')
    configuration['configuration']['instrument']['disperser'] = disperser
    configuration['configuration']['instrument']['filter'] = filt
    #E - I *think* shutter location is just so that the detector gap is placed in the correct place
    configuration['configuration']['instrument']['shutter_location'] = 'q3_345_20'#'q4_345_20'
    # exposure specifications from DEEP APT file
    configuration['configuration']['detector']['ngroup'] = ngroup
    configuration['configuration']['detector']['nint'] = nint
    # PRISM
    configuration['configuration']['detector']['nexp'] = nexp
    configuration['configuration']['detector']['readmode'] = 'nrsirs2'
    configuration['strategy']['dithers'][0]['on_source'] = inputs['onSource']
    configuration['configuration']['instrument']['slitlet_shape'] = inputs['slitletShape']

    #default configuration['configuration'] has 1x3 shutter config

    # --- Build a scene ---
    scene = {}
    if (inputs['re_circ'] > 0):
        sersic = inputs['sersic_n']
        rc = inputs['re_circ']
        ellip = 1. - inputs['axis_ratio']
        pa = inputs['position_angle']

        # pandiea wants scale lengths not half-light radii
        major_axis, minor_axis = majorminor(sersic, rc, ellip)
        scene['position'] = {'x_offset': xoff, 'y_offset': yoff, 'orientation': pa,
                             'position_parameters': ['x_offset', 'y_offset', 'orientation']}
        scene['shape'] = {'geometry': 'sersic', 'sersic_index': sersic,
                          'major': major_axis, 'minor': minor_axis}
    else:
        pa = inputs['position_angle']
        #this is the dummy trigger to go for a point source
        scene['position'] = {'x_offset': xoff, 'y_offset': yoff, 'orientation': pa,
                             'position_parameters': ['x_offset', 'y_offset', 'orientation']}
        scene['shape'] = {'geometry': 'point'}

    scene['spectrum'] = {}
    scene['spectrum']['name'] = "continuum_spectrum"
    # Set redshift to 0 because otherwise it shifts the wavelength array...
    scene['spectrum']['redshift'] = 0
    # FIXME: It doesn't seem to do anything with the normalization of the
    # source spectrum, however?!
    tempIdx = np.where((wl >= filterWlDict[filt]['low']) &
                       (wl <= filterWlDict[filt]['high']))[0]
    scene['spectrum']['sed'] = {'sed_type': 'input',
                                'spectrum': [wl[tempIdx], spec[tempIdx]],
                                'unit': 'mJy'}
    scene['spectrum']['normalization'] = {}
    scene['spectrum']['normalization'] = {'type': 'none'}

    # --- Run pandeia ---
    configuration['scene'][0] = scene
    report = perform_calculation(configuration)

    return report
