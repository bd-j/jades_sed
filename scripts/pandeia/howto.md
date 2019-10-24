# Generate intrinsic spectra

First we use `genspec_parametric` to generate an intrinsic (native library
resolution) spectrum for each object.  This will create an HDF5 file with a group for each object ID and subgroups corresponding to the BEAGLE parameters, the size information, and the intrinsic native resolution fsps spectrum.

```bash
python genspec_parametric.py --add_neb --smoothssp=False --fullspec --outroot=parametric
```

Setting `smoothssp=False --fullspec` ensures that the spectra are at the native library resolution with the full spectral range.  The isochrone and stellar library names will be appended to the end of `outroot`


# Generate observed spectra and S/N estimates

Then we use `fsps_output_to_pandeia` to generate observed spectra (which we will
not use) and S/N estimates as a function of wavelength (which we will use).
These S/N estimates can incorporate size information (and hence slit losses) or
not.  Note that because pandeia does not account for input library spectral resolution when convolving with the line spread funct, the output spectra will be broader than reality and the S/N estimates will be approximate.  Note also that pandeia accounts for undersampling of the LSF in a way that we do not.

```bash
python fsps_output_to_pandeia.py --spectrum_file=parametric_mist_ckc14.h5 --use_sizes --nobj=5000
```


# HDF5 file structure

- 0
  - beagle_parameters (dataset)
  - jaguar_parameters (dataset)
  - prospector_instrinsic (group with attrs)
    - wavelength
    - spectrum
  - DEEP_R100_withSizes (group)
    - wl
    - fnu_noiseless
    - fnu_err
    - fnu
    - sn
  - DEEP_R100 (group, optional)
  - BEAGLE_DEEP_R100 (group, optional)
    - wl
    - fnu_noiseless
    - fnu_err
    - fnu
    - sn

- 1