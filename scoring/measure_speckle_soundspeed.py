# File:       measure_speckle_soundspeed.py
# Author:     Jiaxin Zhang (jzhan295@jhu.edu)
# Created on: 2024-04-17

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from datasets.PWDataLoaders import load_data
from scoring.metrics import snr
from scipy.interpolate import RectBivariateSpline as interp2d
from cubdl.das_torch import DAS_PW
from cubdl.PixelGrid import make_pixel_grid



def measure_speckle_soundspeed(idx):
    snrs = []

    if idx == 0:
        data_source, acq = "TSH", 2
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1
        ROI_idx = 1
        xlims = [-6e-3, 6e-3]
        zlims = [12e-3, 24e-3]
        # ctr_t = [11e-3, 43.3e-3]
        # ctr_b = [5.5e-3, 43.3e-3]
        # roi_size_half = 1e-3
    if idx == 1:
        data_source, acq = "JHU", 28
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # ROI 1
        ROI_idx = 1
        xlims = [-2.5e-3, 2.5e-3]
        zlims = [15e-3, 20e-3]
        # ctr_t = [11e-3, 43.3e-3]
        # ctr_b = [5.5e-3, 43.3e-3]
        # roi_size_half = 1e-3

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
        c_ratio = (P.c / 1540)
        xlims_new = (np.asarray(xlims) * c_ratio).tolist()
        zlims_new = (np.asarray(zlims) * c_ratio).tolist()
        grid = make_pixel_grid(xlims_new, zlims_new, dx, dz)
        fnum = 1

        xext = (np.array([-0.5, grid.shape[1] - 0.5]) * dx + xlims_new[0]) * 1e3
        zext = (np.array([-0.5, grid.shape[0] - 0.5]) * dz + zlims_new[0]) * 1e3
        extent = [xext[0], xext[1], zext[1], zext[0]]

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

        # # Make ROI
        # dist1 = np.abs(grid[:, :, 0] - ctr_t_new[0])
        # dist2 = np.abs(grid[:, :, 2] - ctr_t_new[1])
        # roi_t = (dist1 <= roi_size_half) * (dist2 <= roi_size_half)
        # dist3 = np.abs(grid[:, :, 0] - ctr_b_new[0])
        # dist4 = np.abs(grid[:, :, 2] - ctr_b_new[1])
        # roi_b = (dist3 <= roi_size_half) * (dist4 <= roi_size_half)
        #
        # bimg_t = bimg * roi_t
        # bimg_b = bimg * roi_b
        #
        # bimg_t = bimg * roi_t
        # bimg_b = bimg * roi_b
        # bimg_tt = bimg_t[bimg_t != 0]
        # bimg_bb = bimg_b[bimg_b != 0]

        print("Sound speed = ", P.c)
        print("Speckle SNR = %f" % snr(bimg))

        snrs.append(snr(bimg))

    outdir = os.path.join("snr", "%s%03d" % (data_source, acq))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    hdf5storage.savemat(
        os.path.join(outdir, "roi%02d_c%d-c%d_step%d" % (ROI_idx, sound_speed[0], sound_speed[-1],speed_step)),
        {"snrs": np.stack(snrs)},
    )



if __name__ == "__main__":
    for i in range(1,2):
        measure_speckle_soundspeed(i)