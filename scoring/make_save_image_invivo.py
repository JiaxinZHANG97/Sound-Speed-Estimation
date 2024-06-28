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
    elif idx == 3:
        data_source, acq = "JHU", 24
        image_type = "invivo"
        P, xl, zl = load_data(data_source, acq)
    elif idx == 4:
        data_source, acq = "JHU", 26
        image_type = "invivo"
        P, xl, zl = load_data(data_source, acq)

    P.c = 1539 # save data with this sound speed value

    # Define pixel grid limits (assume y == 0)
    wvln = P.c / P.fc
    dx = wvln / 3
    dz = dx  # Use square pixels
    desired_grid = [400, 300]
    xlims = np.array([-0.5, 0.5]) * (desired_grid[1] - 1) * dx + (xl[0] + xl[1]) / 2
    zlims = np.array([-0.5, 0.5]) * (desired_grid[0] - 1) * dz + (zl[0] + zl[1]) / 2
    xlims = [np.maximum(xlims[0], xl[0]), np.minimum(xlims[1], xl[1])]
    zlims = [np.maximum(zlims[0], zl[0]), np.minimum(zlims[1], zl[1])]
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
        outdir2 = os.path.join("Metadata", moniker, "image", image_type, "%s%03d-c%d" % (data_source, acq, P.c))
        if not os.path.exists(outdir2):
            os.makedirs(outdir2)

        # print(idx)
        print("Start saving Metadata for %s%03d" % (data_source, acq))
        hdf5storage.savemat(
            os.path.join(outdir2, "image_c%d" % P.c),
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

    # Display images via matplotlib
    plt.clf()
    plt.imshow(bimg, vmin=-60, cmap="gray", extent=extent, origin="upper")
    plt.suptitle("%s%03d (c = %d m/s)" % (data_source, acq, np.round(P.c)))
    plt.savefig(os.path.join(outdir2, "image_c%d.png" % (P.c)))
    plt.show()


if __name__ == "__main__":
    for i in range(4,5):
        rois_image_invivo(i)

