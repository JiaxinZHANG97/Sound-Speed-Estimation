# mLOC - CUBDL

## Description

This repository is for coherence-based sound speed estimation [1] in plane wave ultrasound using CUBDL data [2,3]. This method estimates sound speeds by maximizing the maximum short-lag spatial coherence (SLSC) or minimizing the histogram-filtered mean SLSC values within selected ROIs for coherent or incoherent targets, respectively.

This repository uses Python and MATLAB programming languages. CUBDL data are loaded following the same project structure as [CUBDL GitLab](https://gitlab.com/dongwoon.hyun/cubdl/-/tree/master) and are saved as delayed or non-delayed IQ data. IQ data are converted into RF data, which are then used for spatial coherence calculation, mLOC calculation, and max or min SLSC calculation.

P.S.: This method originated from mLOC (maximum lag one coherence) metric [4], which calculates the maximum lag one coherence within a region of interest surrounding a target. This metric was originally designed for sound speed estimation and flexible array shape estimation in photoacoustic imaging.

If you use this code, please cite the following references:

1. Jiaxin Zhang, Yunlong Zhu, and Muyinatu A. Lediju Bell "Coherence-Based Optimization Using Cumulative Spatial Lags to Estimate Sound Speed in Plane Wave Images of Coherent and Incoherent Targets", IUS 2024 [[paper]](https://ieeexplore.ieee.org/abstract/document/10793792)
2. D. Hyun, A. Wiacek, S. Goudarzi, S. Rothlübbers, A. Asif, K. Eickel, Y. C. Eldar, J. Huang, M. Mischi, H. Rivaz, D. Sinden, R.J.G. van Sloun, H. Strohm, M. A. L. Bell, Deep Learning for Ultrasound Image Formation: CUBDL Evaluation Framework & Open Datasets, IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9475029)
3. Muyinatu A. Lediju Bell, Jiaqi Huang, Alycen Wiacek, Ping Gong, Shigao Chen, Alessandro Ramalli, Piero Tortoli, Ben Luijten, Massimo Mischi, Ole Marius Hoel Rindal, Vincent Perrot, Hervé Liebgott, Xi Zhang, Jianwen Luo, Eniola Oluyemi, Emily Ambinder, “Challenge on Ultrasound Beamforming with Deep Learning (CUBDL) Datasets”, IEEE DataPort, 2019 [Online]. Available: http://dx.doi.org/10.21227/f0hn-8f92
4. Jiaxin Zhang, Kai Ding, and Muyinatu A. Lediju Bell "Flexible array curvature and sound speed estimations with a maximum spatial lag-one coherence metric", Proc. SPIE 12842, Photons Plus Ultrasound: Imaging and Sensing 2024, 128421D (12 March 2024); https://doi.org/10.1117/12.3005709

## Notes

### Data loading and saving

#### Save delayed data

Python script [test_submissions_saveIQ.py](submissions/test_submissions_saveIQ.py) loads and saves delayed IQ data of shape `[nsamples, nelements, nplanewaves]`.

Make sure you also have the following Python scripts: [das_torch_saveIQ.py](cubdl/das_torch_saveIQ.py) and [save_IQdata.py](scoring/save_IQdata.py).

Note that this data-saving script takes up lots of memory.

#### Save non-delayed data

Python scripts [make_save_rois_invivo.py](scoring/make_save_rois_invivo.py) and [make_save_image_invivo.py](scoring/make_save_image_invivo.py) load and save non-delayed IQ data for the selected ROI grid or full image grid, respectively. The delay and beamforming steps are done using MATLAB scipts.

Make sure you also have the following Python script: [das_torch_saveData.py](cubdl/das_torch_saveData.py).

### IQ to RF conversion

MATLAB function [iq2rf_jz.m](MATLAB_code/iq2rf_jz.m) converts delayed IQ data to delayed RF data.

Example usage: `rf = iq2rf_jz(I, Q, f0, fs, 1, 1, centerAngle_flag)`

### Spatial coherence and SLSC beamformer

MATLAB function [beamformer_SLSC_PW_US_linear.m](MATLAB_code/beamformer_SLSC_PW_US_linear.m) computes the coherence coefficient matrix and the SLSC image matrix using plane wave data.

Example usage: `[slsc, cc, metadata, x_axis, z_axis] = beamformer_SLSC_PW_US_linear(delay_data, metadata, zero_out_flag)`

Set `zero_out_flag = false` when calculating the max or min SLSC within ROI (e.g., line 142 in mLOC.m). Set `zero_out_flag = true` when plotting SLSC images (e.g., line 165 in CUBDL_SLSC.m).

### DAS and SLSC B-mode image display

MATLAB script [CUBDL_SLSC.m](MATLAB_code/CUBDL_SLSC.m) plots the B-mode images using delay-and-sum beamformer and short-lag spatial-coherence beamformer.

### Maximum or minimum SLSC calculation

MATLAB script [mLOC.m](MATLAB_code/mLOC.m) calculates the maximum and minimum cumulative spatial coherence within the ROI (which is defined in [make_save_rois_invivo.py](scoring/make_save_rois_invivo.py)) per sound speed per M.

### Coherence maps and other plots

MATLAB script [plot_CUBDL_SLSC.m](MATLAB_code/plot_CUBDL_SLSC.m) creates sound speed maps. It plots cumulative spatial coherence (i.e., SLSC) per sound speed per M and looks for the most frequent (i.e., mode) sound speed within M ranging 0-30% of the receiving aperture.
This script also plots speckle brightness, axial FWHM, lateral FWHM, contrast, CNR, and gCNR as functions of sound speed values.

### Image quality metrics

[measure_lesion_soundspeed.py](scoring/measure_lesion_soundspeed.py) measures contrast, CNR, and gCNR of lesion targets at different sound speeds.

[measure_point_soundspeed.py](scoring/measure_point_soundspeed.py) measures lateral and axial FWHM of point targets at different sound speeds.

[measure_speckle_soundspeed.py](scoring/measure_speckle_soundspeed.py) measures speckle SNR of speckle regions at different sound speeds.
