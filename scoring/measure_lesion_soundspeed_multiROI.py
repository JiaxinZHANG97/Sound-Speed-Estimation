# File:       measure_lesion_soundspeed_multiROI.py
# Author:     Jiaxin Zhang (jzhan295@jhu.edu)
# Created on: 2024-10-04

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from datasets.PWDataLoaders import load_data
from scoring.metrics import contrast, cnr, gcnr
from scipy.interpolate import RectBivariateSpline as interp2d
from cubdl.das_torch import DAS_PW
from cubdl.PixelGrid import make_pixel_grid



def measure_lesion_soundspeed(idx):
    contrasts = []
    cnrs = []
    gcnrs = []

    if idx == 0:
        data_source, acq = "INS", 8
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        # ROI_idx = 1
        # xlims = [3.5e-3, 14.5e-3]
        # zlims = [39.5e-3, 47e-3]
        # ctr_t = [11e-3, 43.3e-3]
        # ctr_b = [5.5e-3, 43.3e-3]
        # roi_size_half = 1e-3
        # # ROI 2
        # ROI_idx = 2
        # xlims = [3.5e-3, 14.5e-3]
        # zlims = [39.5e-3, 47e-3]
        # ctr_t = [11.7e-3, 43.5e-3]
        # ctr_b = [5.5e-3, 43.5e-3]
        # roi_size_half = 1e-3
        # # ROI 3
        # ROI_idx = 3
        # c_base = 1540
        # xlims = [3.6e-3, 13.4e-3]
        # zlims = [40.7e-3, 46e-3]
        # ctr_t = [11.5e-3, 43.38e-3]
        # ctr_b = [5.46e-3, 43.38e-3]
        # roi_size_halfX_t = 1.83e-3
        # roi_size_halfZ_t = 2.60e-3
        # roi_size_halfX_b = 1.83e-3
        # roi_size_halfZ_b = 2.60e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 4
        ROI_idx = 4
        c_base = 1540
        xlims = [3.7e-3, 13.1e-3]
        zlims = [40.55e-3, 46.2e-3]
        ctr_t = [11.61e-3, 43.36e-3]
        ctr_b = [5.15e-3, 43.36e-3]
        roi_size_halfX_t = 1.41e-3
        roi_size_halfZ_t = 2.75e-3
        roi_size_halfX_b = 1.41e-3
        roi_size_halfZ_b = 2.75e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 1:
        data_source, acq = "JHU", 28
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1
        # ROI_idx = 1
        # xlims = [-9e-3, 2e-3]
        # zlims = [10e-3, 15e-3]
        # ctr_t = [-0.5e-3, 12.2e-3]
        # ctr_b = [-6.5e-3, 12.2e-3]
        # roi_size_half = 0.5e-3
        # # ROI 2
        # ROI_idx = 2
        # c_base = 1540
        # xlims = [-8.5e-3, 1e-3]
        # zlims = [11e-3, 13.5e-3]
        # ctr_t = [-0.42e-3, 12.1e-3]
        # ctr_b = [-6.66e-3, 12.1e-3]
        # roi_size_halfX = 1.35e-3
        # roi_size_halfZ = 0.8e-3
        # # ROI 3
        ROI_idx = 3
        c_base = 1540
        xlims = [-2e-3, 9.62e-3]
        zlims = [11e-3, 13.5e-3]
        ctr_t = [-0.42e-3, 12.1e-3]
        ctr_b = [8.25e-3, 12.1e-3]
        roi_size_halfX_t = 1.35e-3
        roi_size_halfZ_t = 0.8e-3
        roi_size_halfX_b = 1.35e-3
        roi_size_halfZ_b = 0.8e-3
        num_b_roi = len(ctr_b) / 2
        # # ROI 4
        # ROI_idx = 4
        # c_base = 1540
        # xlims = [-2e-3, 9.5e-3]
        # zlims = [10e-3, 13e-3]
        # ctr_t = [-0.25e-3, 11.45e-3]
        # ctr_b = [7.85e-3, 11.45e-3]
        # roi_size_halfX_t = 1.25e-3
        # roi_size_halfZ_t = 1.05e-3
        # roi_size_halfX_b = 1.25e-3
        # roi_size_halfZ_b = 1.05e-3
        # num_b_roi = len(ctr_b) / 2
    elif idx == 2:
        data_source, acq = "JHU", 27
        P, xl, zl = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1 (based on 1395m/s)
        # ROI_idx = 1
        # c_base = 1395
        # xlims = [-6e-3, 2e-3]
        # zlims = [12e-3, 15e-3]
        # ctr_t = [0.8e-3, 13.5e-3]
        # ctr_b = [-4.2e-3, 13.5e-3]
        # roi_size_halfX = 0.8e-3
        # roi_size_halfZ = 0.5e-3
        # # ROI 2 (based on 1395m/s)
        # ROI_idx = 2
        # c_base = 1395
        # xlims = [-6e-3, 2e-3]
        # zlims = [12e-3, 15e-3]
        # ctr_t = [0.9e-3, 13.3e-3]
        # ctr_b = [-3.5e-3, 13.3e-3]
        # roi_size_halfX_t = 0.8e-3
        # roi_size_halfZ_t = 0.5e-3
        # roi_size_halfX_b = 0.8e-3
        # roi_size_halfZ_b = 0.5e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 3 (P36 mass1 from radiologist ROI3)
        ROI_idx = 3
        c_base = 1540
        ctr_t = [11.9e-3, 14.42e-3]
        ctr_b = [7.19e-3, 14.42e-3]
        roi_size_halfX_t = 0.94e-3
        roi_size_halfZ_t = 0.84e-3
        roi_size_halfX_b = 0.94e-3
        roi_size_halfZ_b = 0.84e-3
        width = xl[1] - xl[0]
        ctr_t[0] = xl[0] + (width - ctr_t[0]) # P36 mass 1 is laterally flipped
        ctr_b[0] = xl[0] + (width - ctr_b[0])  # P36 mass 1 is laterally flipped
        xlims = [-0.2e-3, 6.5e-3]
        zlims = [13.5e-3, 15.3e-3]
        num_b_roi = len(ctr_b) / 2
    elif idx == 3:
        data_source, acq = "JHU", 26
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1
        # ROI_idx = 1
        # c_base = 1540
        # xlims = [-7.5e-3, 1.5e-3]
        # zlims = [13.6e-3, 16e-3]
        # ctr_t = [0e-3, 14.7e-3]
        # ctr_b = [-6e-3, 14.7e-3]
        # roi_size_halfX = 0.9e-3
        # roi_size_halfZ = 0.7e-3
        # # ROI 2
        # ROI_idx = 2
        # xlims = [-7e-3, 1.5e-3]
        # zlims = [13.5e-3, 15.5e-3]
        # ctr_t = [0.4e-3, 14.5e-3]
        # ctr_b = [-5.5e-3, 14.5e-3]
        # roi_size_halfX = 0.9e-3
        # roi_size_halfZ = 0.7e-3
        # # ROI 3
        # ROI_idx = 3
        # xlims = [-7.5e-3, 1.5e-3]
        # zlims = [13.6e-3, 16e-3]
        # ctr_t = [0.05e-3, 14.8e-3]
        # ctr_b = [-5e-3, 14.8e-3]
        # roi_size_halfX = 0.55e-3
        # roi_size_halfZ = 0.49e-3
        # # ROI 4
        # ROI_idx = 4
        # xlims = [-6e-3, 1.5e-3]
        # zlims = [13.6e-3, 16e-3]
        # ctr_t = [0.41e-3, 14.8e-3]
        # ctr_b = [-3.66e-3, 14.8e-3]
        # roi_size_halfX = 0.65e-3
        # roi_size_halfZ = 0.49e-3
        # # ROI 5 (based on 1400m/s)
        # ROI_idx = 5
        # c_base = 1400
        # xlims = [-4.5e-3, 2.3e-3]
        # zlims = [12e-3, 14.5e-3]
        # ctr_t = [0.91e-3, 13.3e-3]
        # ctr_b = [-3.16e-3, 13.3e-3]
        # roi_size_halfX = 0.75e-3
        # roi_size_halfZ = 0.51e-3
        # # ROI 6
        # ROI_idx = 6
        # c_base = 1540
        # xlims = [-7.5e-3, 1.5e-3]
        # zlims = [13.6e-3, 16e-3]
        # ctr_t = [0.05e-3, 15e-3]
        # ctr_b = [-6e-3, 15e-3]
        # roi_size_halfX = 0.9e-3
        # roi_size_halfZ = 0.7e-3
        # # ROI 7
        # ROI_idx = 7
        # c_base = 1540
        # xlims = [-7.5e-3, 1.5e-3]
        # zlims = [13.6e-3, 16e-3]
        # ctr_t = [0.05e-3, 15e-3]
        # ctr_b = [-6e-3, 15e-3]
        # roi_size_halfX = 0.8e-3
        # roi_size_halfZ = 0.6e-3
        # # ROI 8
        # ROI_idx = 8
        # c_base = 1540
        # xlims = [-6.2e-3, 5.4e-3]
        # zlims = [13.5e-3, 17.0e-3]
        # ctr_t = [0.16e-3, 14.75e-3]
        # ctr_b = [-5.42e-3, 14.46e-3, 4.57e-3, 16.05e-3]
        # roi_size_halfX_t = 1.50e-3
        # roi_size_halfZ_t = 0.94e-3
        # roi_size_halfX_b = 0.75e-3
        # roi_size_halfZ_b = 0.94e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 9
        ROI_idx = 9
        c_base = 1540
        xlims = [-6.1e-3, 1.8e-3]
        zlims = [14e-3, 15.6e-3]
        ctr_t = [0.45e-3, 14.82e-3]
        ctr_b = [-4.7e-3, 14.82e-3]
        roi_size_halfX_t = 1.32e-3
        roi_size_halfZ_t = 0.72e-3
        roi_size_halfX_b = 1.32e-3
        roi_size_halfZ_b = 0.72e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 4:
        data_source, acq = "MYO", 1
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        # ROI_idx = 1
        # c_base = 1540
        # xlims = [8.5e-3, 18e-3]
        # zlims = [41e-3, 45e-3]
        # ctr_t = [15.5e-3, 43.3e-3]
        # ctr_b = [10e-3, 43.3e-3]
        # roi_size_halfX = 0.7e-3
        # roi_size_halfZ = 0.6e-3
        # # ROI 2
        # ROI_idx = 2
        # c_base = 1540
        # xlims = [10.1e-3, 16.7e-3]
        # zlims = [42e-3, 44.5e-3]
        # ctr_t = [15.51e-3, 43.26e-3]
        # ctr_b = [11.24e-3, 43.26e-3]
        # roi_size_halfX_t = 1.11e-3
        # roi_size_halfZ_t = 1.17e-3
        # roi_size_halfX_b = 1.11e-3
        # roi_size_halfZ_b = 1.17e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 3
        ROI_idx = 3
        c_base = 1540
        xlims = [8.9e-3, 16.2e-3]
        zlims = [14.8e-3, 17.2e-3]
        ctr_t = [15.13e-3, 16.01e-3]
        ctr_b = [10e-3, 16.01e-3]
        roi_size_halfX_t = 1.05e-3
        roi_size_halfZ_t = 1.17e-3
        roi_size_halfX_b = 1.05e-3
        roi_size_halfZ_b = 1.17e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 5:
        data_source, acq = "INS", 14
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        # ROI_idx = 1
        # c_base = 1540
        # xlims = [-3e-3, 2.5e-3]
        # zlims = [23e-3, 33.5e-3]
        # ctr_t = [0e-3, 25.5e-3]
        # ctr_b = [0e-3, 31e-3]
        # roi_size_halfX = 1.5e-3
        # roi_size_halfZ = 1.4e-3
        # # ROI 2
        # ROI_idx = 2
        # c_base = 1540
        # xlims = [-3e-3, 2.5e-3]
        # zlims = [16e-3, 26e-3]
        # ctr_t = [0e-3, 23.5e-3]
        # ctr_b = [0e-3, 18.5e-3]
        # roi_size_halfX = 1.8e-3
        # roi_size_halfZ = 1.2e-3
        # # ROI 3
        # ROI_idx = 3
        # c_base = 1540
        # xlims = [-3.5e-3, 1.5e-3]
        # zlims = [15e-3, 27e-3]
        # ctr_t = [-1e-3, 24.5e-3]
        # ctr_b = [-1e-3, 17.5e-3]
        # roi_size_halfX = 2e-3
        # roi_size_halfZ = 2e-3
        # # ROI 4
        ROI_idx = 4
        c_base = 1540
        xlims = [-7.5e-3, -2.5e-3]
        zlims = [22e-3, 33.5e-3]
        ctr_t = [-5e-3, 24.5e-3]
        ctr_b = [-5e-3, 31e-3]
        roi_size_halfX_t = 2e-3
        roi_size_halfZ_t = 2e-3
        roi_size_halfX_b = 2e-3
        roi_size_halfZ_b = 2e-3
        num_b_roi = len(ctr_b) / 2
        # # ROI 5
        # ROI_idx = 5
        # c_base = 1540
        # xlims = [-6.5e-3, 2.5e-3]
        # zlims = [22.5e-3, 33.1e-3]
        # ctr_t = [-1.99e-3, 24.54e-3]
        # ctr_b = [-1.99e-3, 30.96e-3]
        # roi_size_halfX_t = 4.41e-3
        # roi_size_halfZ_t = 2.01e-3
        # roi_size_halfX_b = 4.41e-3
        # roi_size_halfZ_b = 2.01e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 6
        # ROI_idx = 6
        # c_base = 1540
        # xlims = [-6.5e-3, 2.5e-3]
        # zlims = [17.2e-3, 31.5e-3]
        # ctr_t = [-1.99e-3, 24.54e-3]
        # ctr_b = [-1.99e-3, 18.27e-3, -1.99e-3, 30.46e-3]
        # roi_size_halfX_t = 4.41e-3
        # roi_size_halfZ_t = 2.01e-3
        # roi_size_halfX_b = 4.41e-3
        # roi_size_halfZ_b = 1.00e-3
        # num_b_roi = len(ctr_b) / 2
    elif idx == 6:
        data_source, acq = "JHU", 29
        P, xl, zl = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1
        # ROI_idx = 1
        # c_base = 1540
        # xlims = [-5e-3, 7.5e-3]
        # zlims = [12e-3, 14e-3]
        # ctr_t = [5.7e-3, 13.15e-3]
        # ctr_b = [-3.2e-3, 13.15e-3]
        # roi_size_halfX = 1.5e-3
        # roi_size_halfZ = 0.65e-3
        # # ROI 2
        # ROI_idx = 2
        # c_base = 1540
        # xlims = [-6e-3, 8e-3]
        # zlims = [12e-3, 14e-3]
        # ctr_t = [5.75e-3, 13.1e-3]
        # ctr_b = [-3.75e-3, 13.1e-3]
        # roi_size_halfX = 1.75e-3
        # roi_size_halfZ = 0.6e-3
        # # ROI 3
        # ROI_idx = 3
        # c_base = 1540
        # xlims = [-7e-3, 8.5e-3]
        # zlims = [12e-3, 14e-3]
        # ctr_t = [5.65e-3, 13e-3]
        # ctr_b = [-4.35e-3, 13e-3]
        # roi_size_halfX = 2.35e-3
        # roi_size_halfZ = 0.7e-3
        # # ROI 4
        # ROI_idx = 4
        # c_base = 1540
        # xlims = [-12.5e-3, 8e-3]
        # zlims = [8.5e-3, 14e-3]
        # ctr_t = [5.87e-3, 11.28e-3]
        # ctr_b = [-10.30e-3, 9.52e-3, -8.07e-3, 11.31e-3, -3.77e-3, 13.06e-3]
        # roi_size_halfX_t = 1.95e-3
        # roi_size_halfZ_t = 2.64e-3
        # roi_size_halfX_b = 1.95e-3
        # roi_size_halfZ_b = 0.88e-3
        # # ROI 5 (P37 mass1 from radiologist ROI)
        ROI_idx = 5
        c_base = 1540
        ctr_t = [13e-3, 7e-3]
        ctr_b = [2.3e-3, 7e-3]
        roi_size_halfX_t = 2.3e-3
        roi_size_halfZ_t = 2.5e-3
        roi_size_halfX_b = 2.3e-3
        roi_size_halfZ_b = 2.5e-3
        ctr_t[0] = xl[0] + ctr_t[0]
        ctr_b[0] = xl[0] + ctr_b[0]
        xlims = [-12.8e-3, 2.8e-3]
        zlims = [4.4e-3, 9.6e-3]


        num_b_roi = len(ctr_b)/2
    elif idx == 7:
        data_source, acq = "JHU", 30
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 2
        ROI_idx = 2
        c_base = 1540
        xlims = [-12.9e-3, 2.7e-3]
        zlims = [7.1e-3, 13.4e-3]
        ctr_t = [0.98e-3, 10.24e-3]
        ctr_b = [-8.35e-3, 11.75e-3, -11.14e-3, 8.73e-3]
        roi_size_halfX_t = 1.61e-3
        roi_size_halfZ_t = 3.02e-3
        roi_size_halfX_b = 1.61e-3
        roi_size_halfZ_b = 1.51e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 8:
        data_source, acq = "JHU", 31
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1
        # ROI_idx = 1
        # c_base = 1540
        # xlims = [-12.7e-3, 12.5e-3]
        # zlims = [4.5e-3, 18e-3]
        # ctr_t = [4.1e-3, 13.87e-3]
        # ctr_b = [11.54e-3, 7.15e-3, -11.71e-3, 10.98e-3, -10.56e-3, 15.67e-3]
        # roi_size_halfX_t = 2.81e-3
        # roi_size_halfZ_t = 2.31e-3
        # roi_size_halfX_b = 0.94e-3
        # roi_size_halfZ_b = 2.31e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 2
        # ROI_idx = 2
        # c_base = 1540
        # xlims = [-12.7e-3, 12.5e-3]
        # zlims = [4.5e-3, 18e-3]
        # ctr_t = [4e-3, 13.95e-3]
        # ctr_b = [-10.92e-3, 6.82e-3, -11.59e-3, 11.28e-3, -10.71e-3, 15.74e-3]
        # roi_size_halfX_t = 2.38e-3
        # roi_size_halfZ_t = 2.13e-3
        # roi_size_halfX_b = 0.79e-3
        # roi_size_halfZ_b = 2.13e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 3
        # ROI_idx = 3
        # c_base = 1540
        # xlims = [-12.7e-3, 12.5e-3]
        # zlims = [4.5e-3, 18e-3]
        # ctr_t = [-4.82e-3, 13.49e-3]
        # ctr_b = [-10.92e-3, 6.82e-3, -11.59e-3, 11.28e-3, -10.71e-3, 15.74e-3]
        # roi_size_halfX_t = 2.38e-3
        # roi_size_halfZ_t = 2.13e-3
        # roi_size_halfX_b = 0.79e-3
        # roi_size_halfZ_b = 2.13e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 4
        # ROI_idx = 4
        # c_base = 1540
        # xlims = [-12.7e-3, 12e-3]
        # zlims = [4.3e-3, 18e-3]
        # ctr_t = [7e-3, 13.56e-3]
        # ctr_b = [9.81e-3, 6.55e-3, -10.07e-3, 5.53e-3, -10.43e-3, 16.80e-3]
        # roi_size_halfX_t = 2.15e-3
        # roi_size_halfZ_t = 3.13e-3
        # roi_size_halfX_b = 2.15e-3
        # roi_size_halfZ_b = 1.04e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 5
        # ROI_idx = 5
        # c_base = 1540
        # xlims = [-12.6e-3, 12.3e-3]
        # zlims = [4.6e-3, 17.9e-3]
        # ctr_t = [4.07e-3, 11.07e-3]
        # ctr_b = [9.46e-3, 6.32e-3, -9.66e-3, 5.50e-3, -9.71e-3, 17.03e-3]
        # roi_size_halfX_t = 2.77e-3
        # roi_size_halfZ_t = 2.28e-3
        # roi_size_halfX_b = 2.77e-3
        # roi_size_halfZ_b = 0.76e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 6
        # ROI_idx = 6
        # c_base = 1540
        # xlims = [-12.6e-3, 12.3e-3]
        # zlims = [4.6e-3, 17.9e-3]
        # ctr_t = [0.08e-3, 10.4e-3]
        # ctr_b = [9.46e-3, 6.32e-3, -9.66e-3, 5.50e-3, -9.71e-3, 17.03e-3]
        # roi_size_halfX_t = 2.77e-3
        # roi_size_halfZ_t = 2.28e-3
        # roi_size_halfX_b = 2.77e-3
        # roi_size_halfZ_b = 0.76e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 7
        # ROI_idx = 7
        # c_base = 1540
        # xlims = [-12.6e-3, 12.1e-3]
        # zlims = [4.4e-3, 18.1e-3]
        # ctr_t = [8.1e-3, 12.97e-3]
        # ctr_b = [8.97e-3, 6.32e-3, -9.28e-3, 5.30e-3, -9.43e-3, 17.24e-3]
        # roi_size_halfX_t = 3.05e-3
        # roi_size_halfZ_t = 2.43e-3
        # roi_size_halfX_b = 3.05e-3
        # roi_size_halfZ_b = 0.81e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 8
        # ROI_idx = 8
        # c_base = 1540
        # xlims = [-12.5e-3, 12e-3]
        # zlims = [4.8e-3, 19.3e-3]
        # ctr_t = [4.74e-3, 13.14e-3]
        # ctr_b = [9.51e-3, 6.42e-3, -9.61e-3, 5.55e-3, -9.97e-3, 18.57e-3, 7.30e-3, 18.47e-3]
        # roi_size_halfX_t = 2.46e-3
        # roi_size_halfZ_t = 2.67e-3
        # roi_size_halfX_b = 2.46e-3
        # roi_size_halfZ_b = 0.67e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 9
        ROI_idx = 9
        c_base = 1540
        xlims = [-12.5e-3, 12e-3]
        zlims = [4.8e-3, 19.3e-3]
        ctr_t = [8.12e-3, 12.98e-3]
        ctr_b = [9.51e-3, 6.42e-3, -9.61e-3, 5.55e-3, -9.97e-3, 18.57e-3, 7.30e-3, 18.47e-3]
        roi_size_halfX_t = 2.46e-3
        roi_size_halfZ_t = 2.67e-3
        roi_size_halfX_b = 2.46e-3
        roi_size_halfZ_b = 0.67e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 9:
        data_source, acq = "JHU", 33
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1
        ROI_idx = 1
        c_base = 1540
        xlims = [-12.3e-3, 7.3e-3]
        zlims = [7.4e-3, 20.1e-3]
        ctr_t = [3.03e-3, 11.15e-3]
        ctr_b = [-9.01e-3, 13.28e-3, 4.01e-3, 18.21e-3]
        roi_size_halfX_t = 3.25e-3
        roi_size_halfZ_t = 3.69e-3
        roi_size_halfX_b = 3.25e-3
        roi_size_halfZ_b = 1.84e-3
        num_b_roi = len(ctr_b) / 2
        # # ROI 2
        ROI_idx = 2
        c_base = 1540
        xlims = [-11.1e-3, 15.1e-3]
        zlims = [8.05e-3, 20.1e-3]
        ctr_t = [3.95e-3, 10.05e-3]
        ctr_b = [-9.12e-3, 13.44e-3, 2.91e-3, 18.11e-3, 13.11e-3, 12.35e-3]
        roi_size_halfX_t = 5.70e-3
        roi_size_halfZ_t = 1.95e-3
        roi_size_halfX_b = 1.90e-3
        roi_size_halfZ_b = 1.95e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 10:
        data_source, acq = "JHU", 34
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1
        # ROI_idx = 1
        # c_base = 1540
        # xlims = [-12.3e-3, 10.4e-3]
        # zlims = [8.5e-3, 19.9e-3]
        # ctr_t = [4.57e-3, 10.90e-3]
        # ctr_b = [-9.38e-3, 13.36e-3, -0.46e-3, 17.44e-3]
        # roi_size_halfX_t = 5.77e-3
        # roi_size_halfZ_t = 2.38e-3
        # roi_size_halfX_b = 2.88e-3
        # roi_size_halfZ_b = 2.38e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 2
        # ROI_idx = 2
        # c_base = 1540
        # xlims = [-11.7e-3, 8.0e-3]
        # zlims = [6.2e-3, 18.7e-3]
        # ctr_t = [5.46e-3, 10.06e-3]
        # ctr_b = [-9.15e-3, 13.36e-3, -0.85e-3, 16.75e-3]
        # roi_size_halfX_t = 2.50e-3
        # roi_size_halfZ_t = 3.84e-3
        # roi_size_halfX_b = 2.50e-3
        # roi_size_halfZ_b = 1.92e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 3
        ROI_idx = 3
        c_base = 1540
        xlims = [-11.9e-3, 6.7e-3]
        zlims = [5.8e-3, 19.5e-3]
        ctr_t = [1.69e-3, 8.1e-3]
        ctr_b = [-9.32e-3, 13.71e-3, -0.41e-3, 17.17e-3]
        roi_size_halfX_t = 4.96e-3
        roi_size_halfZ_t = 2.27e-3
        roi_size_halfX_b = 2.48e-3
        roi_size_halfZ_b = 2.27e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 11:
        data_source, acq = "JHU", 32
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1
        # ROI_idx = 1
        # c_base = 1540
        # xlims = [-12.6e-3, 4.9e-3]
        # zlims = [7.7e-3, 21.7e-3]
        # ctr_t = [-4.25e-3, 11.47e-3]
        # ctr_b = [3.6e-3, 19e-3, -11.23e-3, 10.34e-3]
        # roi_size_halfX_t = 2.57e-3
        # roi_size_halfZ_t = 2.62e-3
        # roi_size_halfX_b = 1.28e-3
        # roi_size_halfZ_b = 2.62e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 2
        # ROI_idx = 2
        # c_base = 1540
        # xlims = [-12.85e-3, 9.6e-3]
        # zlims = [6.1e-3, 20.1e-3]
        # ctr_t = [5.2e-3, 7.65e-3]
        # ctr_b = [3.53e-3, 18.57e-3, -10.61e-3, 10.93e-3]
        # roi_size_halfX_t = 4.38e-3
        # roi_size_halfZ_t = 1.46e-3
        # roi_size_halfX_b = 2.19e-3
        # roi_size_halfZ_b = 1.46e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 3
        ROI_idx = 3
        c_base = 1540
        xlims = [-12.55e-3, 12.3e-3]
        zlims = [3e-3, 19.8e-3]
        ctr_t = [1.57e-3, 10.7e-3]
        ctr_b = [8.55e-3, 4.01e-3, -8.84e-3, 18.74e-3, 4.09e-3, 16.74e-3]
        roi_size_halfX_t = 3.67e-3
        roi_size_halfZ_t = 3.03e-3
        roi_size_halfX_b = 3.67e-3
        roi_size_halfZ_b = 1.01e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 12:
        data_source, acq = "INS", 6
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        # ROI_idx = 1
        # c_base = 1540
        # xlims = [-12.6e-3, -3.1e-3]
        # zlims = [35.7e-3, 40.6e-3]
        # ctr_t = [-10.79e-3, 38.14e-3]
        # ctr_b = [-4.84e-3, 38.14e-3]
        # roi_size_halfX_t = 1.72e-3
        # roi_size_halfZ_t = 2.41e-3
        # roi_size_halfX_b = 1.72e-3
        # roi_size_halfZ_b = 2.41e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 2
        ROI_idx = 2
        c_base = 1540
        xlims = [-6.6e-3, 3.4e-3]
        zlims = [35.7e-3, 40.6e-3]
        ctr_t = [1.60e-3, 38.14e-3]
        ctr_b = [-4.84e-3, 38.14e-3]
        roi_size_halfX_t = 1.72e-3
        roi_size_halfZ_t = 2.41e-3
        roi_size_halfX_b = 1.72e-3
        roi_size_halfZ_b = 2.41e-3
        num_b_roi = len(ctr_b) / 2
        # # ROI 3
        # ROI_idx = 3
        # c_base = 1540
        # xlims = [-6.65e-3, 3.55e-3]
        # zlims = [35.65e-3, 40.5e-3]
        # ctr_t = [1.54e-3, 38.08e-3]
        # ctr_b = [-4.62e-3, 38.08e-3]
        # roi_size_halfX_t = 1.98e-3
        # roi_size_halfZ_t = 2.38e-3
        # roi_size_halfX_b = 1.98e-3
        # roi_size_halfZ_b = 2.38e-3
        # num_b_roi = len(ctr_b) / 2
    elif idx == 13:
        data_source, acq = "INS", 26
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        ROI_idx = 1
        c_base = 1540
        xlims = [-0.8e-3, 1.72e-3]
        zlims = [25.5e-3, 31.6e-3]
        ctr_t = [0.46e-3, 30.68e-3]
        ctr_b = [0.46e-3, 26.47e-3]
        roi_size_halfX_t = 1.25e-3
        roi_size_halfZ_t = 0.89e-3
        roi_size_halfX_b = 1.25e-3
        roi_size_halfZ_b = 0.89e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 14:
        data_source, acq = "INS", 9
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        ROI_idx = 1
        c_base = 1540
        xlims = [-9.3e-3, 19e-3]
        zlims = [33e-3, 42.3e-3]
        ctr_t = [-3.29e-3, 37.67e-3]
        ctr_b = [12.92e-3, 37.67e-3]
        roi_size_halfX_t = 5.98e-3
        roi_size_halfZ_t = 4.60e-3
        roi_size_halfX_b = 5.98e-3
        roi_size_halfZ_b = 4.60e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 15:
        data_source, acq = "MYO", 4
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        ROI_idx = 1
        c_base = 1540
        xlims = [4.3e-3, 13.2e-3]
        zlims = [23.7e-3, 26.1e-3]
        ctr_t = [12.06e-3, 24.90e-3]
        ctr_b = [5.47e-3, 24.90e-3]
        roi_size_halfX_t = 1.13e-3
        roi_size_halfZ_t = 1.17e-3
        roi_size_halfX_b = 1.13e-3
        roi_size_halfZ_b = 1.17e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 16:
        data_source, acq = "OSL", 10
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        # ROI_idx = 1
        # c_base = 1540
        # xlims = [-16.65e-3, 0.7e-3]
        # zlims = [20.5e-3, 27.7e-3]
        # ctr_t = [-8.06e-3, 24.08e-3]
        # ctr_b = [-15.22e-3, 24.08e-3, -0.72e-3, 24.08e-3]
        # roi_size_halfX_t = 2.75e-3
        # roi_size_halfZ_t = 3.54e-3
        # roi_size_halfX_b = 1.38e-3
        # roi_size_halfZ_b = 3.54e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 2
        # ROI_idx = 2
        # c_base = 1540
        # xlims = [-18.7e-3, -5.5e-3]
        # zlims = [20.7e-3, 27.75e-3]
        # ctr_t = [-8.09e-3, 24.22e-3]
        # ctr_b = [-16.10e-3, 24.22e-3]
        # roi_size_halfX_t = 2.58e-3
        # roi_size_halfZ_t = 3.49e-3
        # roi_size_halfX_b = 2.58e-3
        # roi_size_halfZ_b = 3.49e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 3
        # ROI_idx = 3
        # c_base = 1540
        # xlims = [-15.8e-3, -9.5e-3]
        # zlims = [30.2e-3, 43.9e-3]
        # ctr_t = [-12.65e-3, 37.18e-3]
        # ctr_b = [-12.65e-3, 31.53e-3, -12.65e-3, 42.55e-3]
        # roi_size_halfX_t = 3.09e-3
        # roi_size_halfZ_t = 2.58e-3
        # roi_size_halfX_b = 3.098e-3
        # roi_size_halfZ_b = 1.29e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 4
        # ROI_idx = 4
        # c_base = 1540
        # xlims = [-14.85e-3, -8.5e-3]
        # zlims = [30e-3, 44.3e-3]
        # ctr_t = [-11.70e-3, 37.16e-3]
        # ctr_b = [-11.70e-3, 31.47e-3, -11.70e-3, 42.80e-3]
        # roi_size_halfX_t = 3.09e-3
        # roi_size_halfZ_t = 2.89e-3
        # roi_size_halfX_b = 3.09e-3
        # roi_size_halfZ_b = 1.44e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 5
        # ROI_idx = 5
        # c_base = 1540
        # xlims = [-15.4e-3, -8.7e-3]
        # zlims = [29.7e-3, 38.9e-3]
        # ctr_t = [-12.06e-3, 37.09e-3]
        # ctr_b = [-12.06e-3, 31.50e-3]
        # roi_size_halfX_t = 3.29e-3
        # roi_size_halfZ_t = 1.77e-3
        # roi_size_halfX_b = 3.29e-3
        # roi_size_halfZ_b = 1.77e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 6
        ROI_idx = 6
        c_base = 1540
        xlims = [-15.4e-3, -8.7e-3]
        zlims = [35.3e-3, 44.55e-3]
        ctr_t = [-12.06e-3, 37.09e-3]
        ctr_b = [-12.06e-3, 42.73e-3]
        roi_size_halfX_t = 3.29e-3
        roi_size_halfZ_t = 1.77e-3
        roi_size_halfX_b = 3.29e-3
        roi_size_halfZ_b = 1.77e-3
        num_b_roi = len(ctr_b) / 2
    elif idx == 17:
        data_source, acq = "OSL", 7
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        # ROI_idx = 1
        # c_base = 1540
        # xlims = [-10.05e-3, -0.3e-3]
        # zlims = [36.4e-3, 42e-3]
        # ctr_t = [-8.19e-3, 39.23e-3]
        # ctr_b = [-2.15e-3, 39.23e-3]
        # roi_size_halfX_t = 1.81e-3
        # roi_size_halfZ_t = 2.76e-3
        # roi_size_halfX_b = 1.81e-3
        # roi_size_halfZ_b = 2.76e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 2
        # ROI_idx = 2
        # c_base = 1540
        # xlims = [-16.90e-3, -6.35e-3]
        # zlims = [36.4e-3, 42e-3]
        # ctr_t = [-8.19e-3, 39.23e-3]
        # ctr_b = [-15.07e-3, 39.23e-3]
        # roi_size_halfX_t = 1.81e-3
        # roi_size_halfZ_t = 2.76e-3
        # roi_size_halfX_b = 1.81e-3
        # roi_size_halfZ_b = 2.76e-3
        # num_b_roi = len(ctr_b) / 2
        # # ROI 3
        ROI_idx = 3
        c_base = 1540
        xlims = [-17.7e-3, -5.7e-3]
        zlims = [36.8e-3, 41.6e-3]
        ctr_t = [-8.25e-3, 39.27e-3]
        ctr_b = [-15.43e-3, 39.27e-3]
        roi_size_halfX_t = 2.25e-3
        roi_size_halfZ_t = 2.27e-3
        roi_size_halfX_b = 2.25e-3
        roi_size_halfZ_b = 2.27e-3
        num_b_roi = len(ctr_b) / 2
    else:
        raise NotImplementedError

    # Loop through a range of sound speeds
    speed_step = 5
    speed_start = 1350
    speed_end = 1700
    nspeed = int((speed_end - speed_start) / speed_step + 1)
    sound_speed = np.linspace(speed_start, speed_end, num=nspeed)

    for idx in range(nspeed):
        P.c = sound_speed[idx]
        wvln = P.c / P.fc
        dx = wvln / 3
        dz = dx  # Use square pixels
        c_ratio = (P.c / c_base)#.item()
        ctr_t_new = (np.asarray(ctr_t) * c_ratio).tolist()
        ctr_b_new = (np.asarray(ctr_b) * c_ratio).tolist()
        xlims_new = (np.asarray(xlims) * c_ratio).tolist()
        zlims_new = (np.asarray(zlims) * c_ratio).tolist()
        roi_size_halfX_t_new = roi_size_halfX_t * c_ratio
        roi_size_halfZ_t_new = roi_size_halfZ_t * c_ratio
        roi_size_halfX_b_new = roi_size_halfX_b * c_ratio
        roi_size_halfZ_b_new = roi_size_halfZ_b * c_ratio
        grid = make_pixel_grid(xlims_new, zlims_new, dx, dz)
        fnum = 1

        xext = (np.array([-0.5, grid.shape[1] - 0.5]) * dx + xlims_new[0]) * 1e3
        zext = (np.array([-0.5, grid.shape[0] - 0.5]) * dz + zlims_new[0]) * 1e3
        extent = [xext[0], xext[1], zext[1], zext[0]]

        outdir = os.path.join("contrast-cnr-gcnr", "%s%03d" % (data_source, acq))
        fname = os.path.join(outdir, "roi%02d_c%d-c%d_step%d_ROIupdate_multi" % (
        ROI_idx, sound_speed[0], sound_speed[-1], speed_step))
        if os.path.exists(fname):
            print("%s exists. Skipping..." % fname)
            continue
        else:
            print("Processing %s..." % fname)

        # Normalize input to [-1, 1] range
        maxval = np.maximum(np.abs(P.idata).max(), np.abs(P.qdata).max())
        P.idata /= maxval
        P.qdata /= maxval

        # Make data torch tensors
        x = (P.idata, P.qdata)

        das = DAS_PW(P, grid, rxfnum=fnum)

        idas, qdas = das(x)
        idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
        iq = idas + 1j * qdas
        bimg = np.abs(iq)
        bimg /= np.amax(bimg)

        plt.clf()
        plt.imshow(20 * np.log10(bimg), vmin=-40, cmap="gray", extent=extent, origin="upper")
        plt.suptitle("%s%03d (c = %d m/s)" % (data_source, acq, P.c))
        plt.show()

        # Make target ROI
        dist1 = np.abs(grid[:, :, 0] - ctr_t_new[0])
        dist2 = np.abs(grid[:, :, 2] - ctr_t_new[1])
        roi_t = (dist1 <= roi_size_halfX_t_new) * (dist2 <= roi_size_halfZ_t_new)
        # Make background ROI
        roi_b = np.zeros_like(roi_t)
        for r in range(int(num_b_roi)):
            dist3 = np.abs(grid[:, :, 0] - ctr_b_new[r*2+0])
            dist4 = np.abs(grid[:, :, 2] - ctr_b_new[r*2+1])
            roi_b_temp = (dist3 <= roi_size_halfX_b_new) * (dist4 <= roi_size_halfZ_b_new)
            roi_b = roi_b + roi_b_temp

        bimg_t = bimg * roi_t
        bimg_b = bimg * roi_b
        bimg_tt = bimg_t[bimg_t != 0]
        bimg_bb = bimg_b[bimg_b != 0]

        print("Sound speed = ", P.c)
        print("Contrast = %f dB" % contrast(bimg_tt, bimg_bb))
        print("CNR = %f" % cnr(bimg_tt, bimg_bb))
        print("gCNR = %f" % gcnr(bimg_tt, bimg_bb))

        contrasts.append(contrast(bimg_tt, bimg_bb))
        cnrs.append(cnr(bimg_tt, bimg_bb))
        gcnrs.append(gcnr(bimg_tt, bimg_bb))

    # outdir = os.path.join("contrast-cnr-gcnr", "%s%03d" % (data_source, acq))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    hdf5storage.savemat(
        fname,
        {
            "c_range": np.array([speed_start,speed_end]),
            "contrasts": np.stack(contrasts),
            "cnrs": np.stack(cnrs),
            "gcnrs": np.stack(gcnrs),
        },
    )



if __name__ == "__main__":
    for i in range(6,7):
        measure_lesion_soundspeed(i)