# mLOC - CUBDL

## Description

This repository is for applying the mLOC metric [1] to the CUBDL data [2,3]. mLOC calculates the maximum lag one coherence within a region of interest surrounding a target. This metric was originally designed for sound speed estimation and flexible array shape estimation in photoacoustic imaging.

This repository uses Python and MATLAB programming languages. CUBDL data are loaded following the same project structure as {CUBDL GitLab](https://gitlab.com/dongwoon.hyun/cubdl/-/tree/master) and are saved as delayed IQ data. IQ data are converted into RF data, which are then used for spatial coherence calculation and mLOC calculation.

If you use this code, please cite the following three references:
 
1. Jiaxin Zhang, Kai Ding, and Muyinatu A. Lediju Bell "Flexible array curvature and sound speed estimations with a maximum spatial lag-one coherence metric", Proc. SPIE 12842, Photons Plus Ultrasound: Imaging and Sensing 2024, 128421D (12 March 2024); https://doi.org/10.1117/12.3005709
2. D. Hyun, A. Wiacek, S. Goudarzi, S. Rothlübbers, A. Asif, K. Eickel, Y. C. Eldar, J. Huang, M. Mischi, H. Rivaz, D. Sinden, R.J.G. van Sloun, H. Strohm, M. A. L. Bell, Deep Learning for Ultrasound Image Formation: CUBDL Evaluation Framework & Open Datasets, IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9475029)
3. Muyinatu A. Lediju Bell, Jiaqi Huang, Alycen Wiacek, Ping Gong, Shigao Chen, Alessandro Ramalli, Piero Tortoli, Ben Luijten, Massimo Mischi, Ole Marius Hoel Rindal, Vincent Perrot, Hervé Liebgott, Xi Zhang, Jianwen Luo, Eniola Oluyemi, Emily Ambinder, “Challenge on Ultrasound Beamforming with Deep Learning (CUBDL) Datasets”, IEEE DataPort, 2019 [Online]. Available: http://dx.doi.org/10.21227/f0hn-8f92

## Notes

### Data loading and saving

Python script [test_submissions_saveIQ.py](submissions/test_submissions_saveIQ.py) loads and saves delayed IQ data of shape `[nsamples, nelements, nplanewaves]`.

Make sure you also have the following Python scripts: [das_torch_saveIQ.py](cubdl/das_torch_saveIQ.py) and [save_IQdata.py](scoring/save_IQdata.py).

Note that this data-saving script takes up lots of memory.

### IQ to RF conversion

MATLAB function [iq2rf_jz.m](MATLAB code/iq2rf_jz.m) converts delayed IQ data to delayed RF data.

Example usage: `rf = iq2rf_jz(I, Q, f0, fs, 1, 1, centerAngle_flag)`

### Spatial coherence and SLSC beamformer

MATLAB function [beamformer_SLSC_PW_US_linear.m](MATLAB code/beamformer_SLSC_PW_US_linear.m) computes the coherence coefficient matrix and the SLSC image matrix using plane wave data.

Example usage: `[slsc, cc, metadata, x_axis, z_axis] = beamformer_SLSC_US_linear(delay_data, metadata)`

### DAS and SLSC B-mode image display

MATLAB script [CUBDL_SLSC.m](MATLAB_code/CUBDL_SLSC.m) plots the B-mode images using delay-and-sum beamformer and short-lag spatial-coherence beamformer.

### mLOC calculation

MATLAB script [mLOC.m](MATLAB code/mLOC.m) calculates the lag one coherence, selects a region of interest surrounding a target, and finds the maximum spatial coherence.

This script also computes 6 image quality metrics, including lateral FWHM, axial FWHM, contrast, CNR, SNR, and gCNR.


