# File:       make_rois_image_invivo.py
# Author:     Dongwoon Hyun (dongwoon.hyun@stanford.edu)
# Created on: 2020-07-31
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from datasets.PWDataLoaders import load_data
from cubdl.PixelGrid import make_pixel_grid
from cubdl.das_torch import DAS_PW
from cubdl.das_torch_saveIQ import DAS_PW_saveIQ
from cubdl.das_torch_saveData import DAS_PW_saveData
import hdf5storage


def rois_image_invivo(idx, outdir=os.path.join("scoring","rois","image","invivo")):
    if idx == 0:
        data_source, acq = "EUT", 1
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
    elif idx == 1:
        data_source, acq = "INS", 8
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
    elif idx == 2:
        data_source, acq = "JHU", 27
        image_type = "invivo"
        P, xl, zl = load_data(data_source, acq)
        xl = [-7.5e-3, 7.5e-3]
        zl = [5e-3, 25e-3]
    elif idx == 3:
        data_source, acq = "JHU", 24
        image_type = "invivo"
        P, xl, zl = load_data(data_source, acq)
        xl = [-8e-3, 5e-3]
        zl = [7e-3, 25e-3]
    elif idx == 4:
        data_source, acq = "JHU", 26
        image_type = "invivo"
        P, xl, zl = load_data(data_source, acq)
        xl = [-7.5e-3, 7.5e-3]
        zl = [5e-3, 25e-3]
        # xl = [-5e-3, 5e-3]
        # zl = [10e-3, 20e-3]
        xl = [-7.5e-3, 7.5e-3]
        zl = [5e-3, 22e-3]
    elif idx == 5:
        data_source, acq = "INS", 12
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        xl = [-12e-3, 17e-3]
        zl = [15e-3, 50e-3]
    elif idx == 6:
        data_source, acq = "MYO", 1
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        xl = [10e-3, xl[1]]
        zl = [11e-3, 20e-3]
        # xshift = 15e-3
        # zshift = 10e-3
    elif idx == 7:
        data_source, acq = "INS", 21
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
    elif idx == 8:
        data_source, acq = "INS", 14
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        # xl = [-3e-3,11e-3]
        # zl = [15e-3,36e-3]
        xl = [-10e-3,12e-3]
        zl = [16e-3,50e-3]
    elif idx == 9:
        data_source, acq = "JHU", 29
        image_type = "invivo"
        P, xl, zl = load_data(data_source, acq)
        # xl = [-3e-3,11e-3]
        zl = [3e-3, 27e-3]
    elif idx == 10:
        data_source, acq = "JHU", 30
        image_type = "invivo"
        P, xl, zl = load_data(data_source, acq)
        zl = [4e-3, 17e-3]
    elif idx == 11:
        data_source, acq = "JHU", 31
        image_type = "invivo"
        P, xl, zl = load_data(data_source, acq)
        # zl = [4e-3, 18e-3]
    elif idx == 12:
        data_source, acq = "JHU", 33
        image_type = "invivo"
        P, xl, zl = load_data(data_source, acq)
        xl = [-16e-3, 15e-3]
        zl = [3.5e-3, 20e-3]
    elif idx == 13:
        data_source, acq = "JHU", 32
        image_type = "invivo"
        P, xl, zl = load_data(data_source, acq)
        # xl = [-7.5e-3, 7.5e-3]
        zl = [3e-3, 23e-3]
    elif idx == 14:
        data_source, acq = "INS", 6
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        # # xl = [-7.5e-3, 7.5e-3]
        zl = [30e-3, 45e-3]
    elif idx == 15:
        data_source, acq = "INS", 26
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        # # xl = [-7.5e-3, 7.5e-3]
        zl = [24e-3, 35e-3]
    elif idx == 16:
        data_source, acq = "EUT", 6
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        # # xl = [-7.5e-3, 7.5e-3]
        zl = [17e-3, 29e-3]
    elif idx == 17:
        data_source, acq = "INS", 9
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        xl = [-15e-3, 18.9e-3]
        zl = [29e-3, 47e-3]
    elif idx == 18:
        data_source, acq = "MYO", 4
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        xl = [2e-3, 16e-3]
        zl = [20e-3, 30e-3]
    elif idx == 19:
        data_source, acq = "OSL",7
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        xl = [-18e-3, 0.5e-3]
        zl = [34e-3, 43e-3]
    elif idx == 20:
        data_source, acq = "OSL",10
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        xl = [-18.8e-3, 9.5e-3]
        zl = [12e-3, 44.6e-3]
    elif idx == 21:
        data_source, acq = "OSL",3
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        xl = [-8e-3, 3e-3]
        zl = [15e-3, 25e-3]
    elif idx == 22:
        data_source, acq = "INS", 25
        image_type = "phantom"
        P, xl, zl = load_data(data_source, acq)
        zl = [10e-3, 39e-3]
    elif idx == 23:
        data_source, acq = "JHU", 28
        image_type = "invivo"
        P, xl, zl = load_data(data_source, acq)
        xl = [-10e-3, 10e-3]
        zl = [2.5e-3, 28e-3]

    P.c = 1540 # save data with this sound speed value

    # Define pixel grid limits (assume y == 0)
    wvln = P.c / P.fc
    dx = wvln / 3 # Original
    # dx = wvln / 2  # Larger pixel
    dz = dx  # Use square pixels

    ## New grid (JZ)
    xlims = xl
    zlims = zl

    grid = make_pixel_grid(xlims, zlims, dx, dz)
    fnum = 1

    xext = (np.array([-0.5, grid.shape[1] - 0.5]) * dx + xlims[0]) * 1e3
    zext = (np.array([-0.5, grid.shape[0] - 0.5]) * dz + zlims[0]) * 1e3
    extent = [xext[0], xext[1], zext[1], zext[0]]

    # Normalize input to [-1, 1] range
    maxval = np.maximum(np.abs(P.idata).max(), np.abs(P.qdata).max())
    P.idata /= maxval
    P.qdata /= maxval

    # Make data torch tensors
    x = (P.idata, P.qdata)

    # Save delayed IQ data or not delayed data
    saveDelayed = False

    # Make 75-angle image
    if saveDelayed:
        das = DAS_PW_saveIQ(P, grid, rxfnum=fnum)
        idas, qdas, idelayed_pw, qdelayed_pw = das(x)
        idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
        idelayed_pw, qdelayed_pw = idelayed_pw.detach().cpu().numpy(), qdelayed_pw.detach().cpu().numpy()

        iq = idas + 1j * qdas
        bimg = 20 * np.log10(np.abs(iq))  # Log-compress
        bimg -= np.amax(bimg)  # Normalize by max value

        moniker = "ground_truth"
        center_angle = False
        outdir2 = os.path.join("IQdata", moniker, "roi", "%s%03d" % (data_source, acq))
        if not os.path.exists(outdir2):
            os.makedirs(outdir2)

        ### Save delayed IQ data ###
        print(idx)
        print("Start saving IQ data for %s%03d" % (data_source, acq))
        hdf5storage.savemat(
            os.path.join(outdir2, "roi%02d_c%d" % (idx, P.c)),
            {
                "idelayed_pw": idelayed_pw,
                "qdelayed_pw": qdelayed_pw,
                'f0': P.fc,
                'fs': P.fs,
                'c': P.c,
                'acq': acq,
                'data_source': data_source,
                'extent': extent,
                "moniker": moniker,
                'center_angle': center_angle,
                'bimg_shape': bimg.shape
            },
        )
        print("End saving IQ data")

    else:
        das = DAS_PW_saveData(P, grid, rxfnum=fnum)
        idas, qdas, txapo, rxapo, txdel, rxdel = das(x)
        idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
        txapo, rxapo = txapo.detach().cpu().numpy(), rxapo.detach().cpu().numpy()
        txdel, rxdel = txdel.detach().cpu().numpy(), rxdel.detach().cpu().numpy()

        iq = idas + 1j * qdas
        bimg = 20 * np.log10(np.abs(iq))  # Log-compress
        bimg -= np.amax(bimg)  # Normalize by max value

        ### Save not delayed IQ data ###
        moniker = "ground_truth"
        center_angle = False
        outdir2 = os.path.join("Metadata", moniker, "image", image_type, "%s%03d" % (data_source, acq), "%s%03d-c%d" % (data_source, acq, P.c))
        if not os.path.exists(outdir2):
            os.makedirs(outdir2)
        #
        print(idx)
        print("Start saving Metadata for %s%03d" % (data_source, acq))
        hdf5storage.savemat(
            os.path.join(outdir2, "image_c%d_portion_largepixel" % P.c),
            {
                "txapo": txapo,
                "rxapo": rxapo,
                "txdel": txdel,
                "rxdel": rxdel,
                'f0': P.fc,
                'fs': P.fs,
                'c': P.c,
                'idata': P.idata,
                'qdata': P.qdata,
                'acq': acq,
                'data_source': data_source,
                'extent': extent,
                "moniker": moniker,
                'center_angle': center_angle,
                'bimg_shape': bimg.shape
            },
        )
        print("End saving Metadata")

    ratio = 1#0.7
    portion = np.round((ratio * np.shape(bimg)[0]))
    portion = portion.astype(int)
    extent_portion = extent.copy()
    extent_portion[2]=(extent_portion[2]-extent_portion[3])*ratio+extent_portion[3]

    # Display images via matplotlib
    plt.clf()
    plt.imshow(bimg[0:portion,:], vmin=-60, cmap="gray", extent=extent_portion, origin="upper")
    plt.suptitle("%s%03d (c = %d m/s)" % (data_source, acq, np.round(P.c)))
    plt.xlabel("Lateral (mm)", fontsize=14)
    plt.ylabel("Axial (mm)", fontsize=14)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.savefig(os.path.join(outdir2, "image_c%d_font.png" % (P.c)))
    plt.show()


if __name__ == "__main__":
    for i in range(2,3):
        rois_image_invivo(i)

