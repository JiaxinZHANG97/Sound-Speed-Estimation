# File:       measure_lesion_soundspeed.py
# Author:     Jiaxin Zhang (jzhan295@jhu.edu)
# Created on: 2024-04-17

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
        ROI_idx = 3
        c_base = 1540
        xlims = [3.6e-3, 13.4e-3]
        zlims = [40.7e-3, 46e-3]
        ctr_t = [11.5e-3, 43.38e-3]
        ctr_b = [5.46e-3, 43.38e-3]
        roi_size_halfX = 1.83e-3
        roi_size_halfZ = 2.60e-3
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
        # ROI_idx = 3
        # c_base = 1540
        # xlims = [-2e-3, 9.62e-3]
        # zlims = [11e-3, 13.5e-3]
        # ctr_t = [-0.42e-3, 12.1e-3]
        # ctr_b = [8.25e-3, 12.1e-3]
        # roi_size_halfX = 1.35e-3
        # roi_size_halfZ = 0.8e-3
        # # ROI 4
        ROI_idx = 4
        c_base = 1540
        xlims = [-2e-3, 9.5e-3]
        zlims = [10e-3, 13e-3]
        ctr_t = [-0.25e-3, 11.45e-3]
        ctr_b = [7.85e-3, 11.45e-3]
        roi_size_halfX = 1.25e-3
        roi_size_halfZ = 1.05e-3
    elif idx == 2:
        data_source, acq = "JHU", 27
        P, _, _ = load_data(data_source, acq)
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
        # roi_size_halfX = 0.8e-3
        # roi_size_halfZ = 0.5e-3
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
        ROI_idx = 7
        c_base = 1540
        xlims = [-7.5e-3, 1.5e-3]
        zlims = [13.6e-3, 16e-3]
        ctr_t = [0.05e-3, 15e-3]
        ctr_b = [-6e-3, 15e-3]
        roi_size_halfX = 0.8e-3
        roi_size_halfZ = 0.6e-3
    elif idx == 4:
        data_source, acq = "MYO", 1
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        ROI_idx = 1
        c_base = 1540
        xlims = [8.5e-3, 18e-3]
        zlims = [41e-3, 45e-3]
        ctr_t = [15.5e-3, 43.3e-3]
        ctr_b = [10e-3, 43.3e-3]
        roi_size_halfX = 0.7e-3
        roi_size_halfZ = 0.6e-3
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
        roi_size_halfX = 2e-3
        roi_size_halfZ = 2e-3
    elif idx == 6:
        data_source, acq = "JHU", 29
        P, _, _ = load_data(data_source, acq)
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
    elif idx == 7:
        data_source, acq = "JHU", 30
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1
        ROI_idx = 1
        c_base = 1540
        xlims = [-12.5e-3, 4e-3]
        zlims = [9e-3, 13.5e-3]
        ctr_t = [1.29e-3, 11.26e-3]
        ctr_b = [-9.78e-3, 11.26e-3]
        roi_size_halfX = 2.43e-3
        roi_size_halfZ = 2e-3
    elif idx == 8:
        data_source, acq = "JHU", 24
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1
        ROI_idx = 1
        c_base = 1540
        xlims = [-4.3e-3, 2.9e-3]
        zlims = [10e-3, 12e-3]
        ctr_t = [-2.77e-3, 10.93e-3]
        ctr_b = [1.19e-3, 10.93e-3]
        roi_size_halfX = 1.25e-3
        roi_size_halfZ = 0.60e-3
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
        roi_size_halfX_new = roi_size_halfX * c_ratio
        roi_size_halfZ_new = roi_size_halfZ * c_ratio
        grid = make_pixel_grid(xlims_new, zlims_new, dx, dz)
        fnum = 1

        xext = (np.array([-0.5, grid.shape[1] - 0.5]) * dx + xlims_new[0]) * 1e3
        zext = (np.array([-0.5, grid.shape[0] - 0.5]) * dz + zlims_new[0]) * 1e3
        extent = [xext[0], xext[1], zext[1], zext[0]]

        outdir = os.path.join("contrast-cnr-gcnr", "%s%03d" % (data_source, acq))
        fname = os.path.join(outdir, "roi%02d_c%d-c%d_step%d_ROIupdate" % (
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

        # Make ROI
        dist1 = np.abs(grid[:, :, 0] - ctr_t_new[0])
        dist2 = np.abs(grid[:, :, 2] - ctr_t_new[1])
        roi_t = (dist1 <= roi_size_halfX_new) * (dist2 <= roi_size_halfZ_new)
        dist3 = np.abs(grid[:, :, 0] - ctr_b_new[0])
        dist4 = np.abs(grid[:, :, 2] - ctr_b_new[1])
        roi_b = (dist3 <= roi_size_halfX_new) * (dist4 <= roi_size_halfZ_new)

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
    for i in range(2,3):
        measure_lesion_soundspeed(i)