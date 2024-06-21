# File:       measure_point_soundspeed.py
# Author:     Jiaxin Zhang (jzhan295@jhu.edu)
# Created on: 2024-04-16

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import hdf5storage
from datasets.PWDataLoaders import load_data
from scoring.metrics import res_FWHM
from scipy.interpolate import RectBivariateSpline as interp2d
from cubdl.das_torch import DAS_PW
from cubdl.PixelGrid import make_pixel_grid



def measure_point_soundspeed(idx):
    x_fwhms = []
    z_fwhms = []
    bimgs = []
    exts = []

    if idx == 0:
        data_source, acq = "EUT", 1
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        # ROI_idx = 1
        # xlims = [-4.5e-3, -0.5e-3]
        # zlims = [18e-3, 22e-3]
        # # ROI 2
        # ROI_idx = 2
        # xlims = [0e-3, 5e-3]
        # zlims = [18e-3, 22e-3]
        # # ROI 3
        # ROI_idx = 3
        # xlims = [1e-3, 6e-3]
        # zlims = [13e-3, 17e-3]
        # # ROI 4
        ROI_idx = 4
        xlims = [-4.5e-3, -0.5e-3]
        zlims = [27e-3, 31e-3]

    else:
        raise NotImplementedError

    # Loop through a range of sound speeds
    speed_step = 5
    speed_start = 1400
    speed_end = 1700
    nspeed = int((speed_end - speed_start) / speed_step + 1)
    sound_speed = np.linspace(speed_start, speed_end, num=nspeed)

    for idx in range(nspeed):
        P.c = sound_speed[idx]
        wvln = P.c / P.fc
        dx = wvln / 3
        dz = dx  # Use square pixels
        c_ratio = (P.c / 1540).item()
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

        ax = grid[:, 0, 2]  # Axial pixel positions [m]
        az = grid[0, :, 0]  # Azimuthal pixel positions [m]
        f = interp2d(ax, az, bimg)

        bimg_max_idx = np.unravel_index(np.argmax(bimg, axis=None), bimg.shape)
        ax_max = ax[bimg_max_idx[0]]
        az_max = az[bimg_max_idx[1]]
        pt = [az_max, 0, ax_max]

        # Define pixel grid limits (assume y == 0)
        dp = P.c / P.fc / 3 / 100  # Interpolate 100-fold
        roi = np.arange(-1e-3, 1e-3, dp)
        roi -= np.mean(roi)
        zeros = np.zeros_like(roi)
        # Horizontal grid
        xroi = f(pt[2] + zeros, pt[0] + roi)[0]
        zroi = f(pt[2] + roi, pt[0] + zeros)[:, 0]

        x_fwhm = res_FWHM(xroi) * dp * 1e3
        z_fwhm = res_FWHM(zroi) * dp * 1e3
        print("Sound speed = ", P.c)
        print("X FWHM = %f mm" % x_fwhm)
        print("Z FWHM = %f mm" % z_fwhm)

        x_fwhms.append(x_fwhm)
        z_fwhms.append(z_fwhm)

    outdir = os.path.join("FWHM", "%s%03d" % (data_source, acq))
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    hdf5storage.savemat(
        os.path.join(outdir, "roi%02d_c%d-c%d_step%d" % (ROI_idx, sound_speed[0], sound_speed[-1],speed_step)),
        {"x_fwhms": np.stack(x_fwhms), "z_fwhms": np.stack(z_fwhms)},
    )



if __name__ == "__main__":
    for i in range(1):
        measure_point_soundspeed(i)