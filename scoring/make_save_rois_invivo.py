# File:       make_save_rois_invivo.py
# Author:     Jiaxin Zhang (jzhan295@jhu.edu)
# Created on: 2024-03-27
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

device = torch.device("cuda:0")


def rois_invivo(idx, outdir=os.path.join("scoring", "rois", "invivo")):
    if idx == 0:
        data_source, acq = "TSH", 2
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # roi1
        # xlims = [-8.5e-3, -2.5e-3]
        # zlims = [24.5e-3, 30.5e-3]
        # roi2
        # xlims = [-12.3e-3, -7.3e-3]
        # zlims = [34e-3, 39e-3]
        # roi3
        # xlims = [-12.3e-3, -7.3e-3]
        # zlims = [35e-3, 43e-3]
        # roi4
        # xlims = [-12.3e-3, -7.3e-3]
        # zlims = [11.4e-3, 16.4e-3]
        # roi5
        # xlims = [-8.5e-3, -2.5e-3]
        # zlims = [24e-3, 31e-3]
        # roi6
        # roi = 6
        # xlims = [7.3e-3, 12.3e-3]
        # zlims = [20e-3, 25e-3]
        # roi7
        # roi = 7
        # xlims = [-7e-3, -4e-3]
        # zlims = [25e-3, 28e-3]
        # ROI 11
        # roi = 11
        # xlims = [-12.3e-3, -5.3e-3]
        # zlims = [13e-3, 20e-3]
        # # ROI 12
        # roi = 12
        # xlims = [6.5e-3, 11.5e-3]
        # zlims = [20e-3, 25e-3]
        # # ROI 13
        roi = 13
        xlims = [6.5e-3, 12e-3]
        zlims = [20e-3, 25e-3]

    elif idx == 1:
        data_source, acq = "JHU", 28
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # roi1
        # xlims = [-5e-3, 5e-3]
        # zlims = [4e-3, 18e-3]
        # # roi2
        # xlims = [-2.5e-3, 2.5e-3]
        # zlims = [7e-3, 15e-3]
        # # # roi3 (lesion)
        # roi = 3
        # xlims = [-2e-3, 1e-3]
        # zlims = [10.5e-3, 13.5e-3]
        # # roi4 (fat)
        # xlims = [-1.5e-3, 1.5e-3]
        # zlims = [15e-3, 18e-3]
        # roi5 (fat)
        # xlims = [-4e-3, 0e-3]
        # zlims = [17.5e-3, 21.5e-3]
        # roi8 (fat)
        # roi = 8
        # xlims = [0e-3, 5e-3]
        # zlims = [15e-3, 20e-3]
        # # roi13 (lesion)
        roi = 13
        xlims = [-2.5e-3, 1.5e-3]
        zlims = [10e-3, 14e-3]


    elif idx == 2:
        data_source, acq = "EUT", 1
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        # roi = 1
        # xlims = [2e-3, 5e-3]
        # zlims = [13e-3, 17e-3]
        # # ROI 2
        # roi = 2
        # xlims = [-4.5e-3, -0.5e-3]
        # zlims = [18e-3, 22e-3]
        # ROI 3
        roi = 3
        xlims = [-6e-3, -1e-3]
        zlims = [35.5e-3, 40.5e-3]

    elif idx == 3:
        data_source, acq = "INS", 8
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # roi 1
        # roi = 1
        # xlims = [7e-3, 13e-3]
        # zlims = [41e-3, 47e-3]
        # # roi 2
        # roi = 2
        # xlims = [7e-3, 13.5e-3]
        # zlims = [39.5e-3, 46.5e-3]
        # roi 3
        roi = 3
        xlims = [7e-3, 14.7e-3]
        zlims = [39.5e-3, 46.5e-3]

    elif idx == 4:
        data_source, acq = "JHU", 27 # patient breast mass
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1
        roi = 1
        xlims = [-0.2e-3, 2.5e-3]
        zlims = [14.4e-3, 16e-3]

    elif idx == 5:
        data_source, acq = "JHU", 24
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1
        roi = 1
        xlims = [-4.5e-3, -2e-3]
        zlims = [9e-3, 13e-3]
        # # roi 2
        # roi = 2
        # xlims = [3.4e-3, 7.7e-3]
        # zlims = [13.1e-3, 16.5e-3]
    elif idx == 6:
        data_source, acq = "JHU", 26
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1
        roi = 1
        xlims = [-2e-3, 1.5e-3]
        zlims = [13.5e-3, 16.5e-3]
        # # roi 2
        # roi = 2
        # xlims = [-2.4e-3, 1.6e-3]
        # zlims = [13.5e-3, 16.5e-3]
    elif idx == 7:
        data_source, acq = "INS", 14
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # roi 1 (point target)
        roi = 1
        xlims = [7.1e-3, 10.7e-3]
        zlims = [33.3e-3, 35.3e-3]
        # # roi 2 (vessel)
        # roi = 2
        # xlims = [-6.5e-3, -3.5e-3]
        # zlims = [23e-3, 26e-3]
    elif idx == 8:
        data_source, acq = "INS", 12
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # roi 1 (point target)
        roi = 1
        xlims = [10.5e-3, 14.8e-3]
        zlims = [30.2e-3, 32.3e-3]
    else:
        raise NotImplementedError

    # Save delayed IQ data or not delayed data
    saveDelayed = False

    # Loop through a range of sound speeds
    speed_step = 5
    speed_start = 1350
    speed_end = 1700
    nspeed = int((speed_end - speed_start)/speed_step+1)
    sound_speed = np.linspace(speed_start,speed_end, num=nspeed)

    speckle_brightness = np.zeros(len(sound_speed))

    for idx in range(nspeed): # range(17,21): range(nspeed):
        P.c = sound_speed[idx]
        wvln = P.c / P.fc
        dx = wvln / 3
        dz = dx  # Use square pixels
        c_ratio = (P.c / 1540).item()
        xlims_new = (np.asarray(xlims) * c_ratio).tolist()
        zlims_new = (np.asarray(zlims)*c_ratio).tolist()
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
            outdir2 = os.path.join("IQdata", moniker, "roi", image_type, "%s%03d" % (data_source, acq))
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
            outdir2 = os.path.join("Metadata", moniker, "roi", image_type, "%s%03d_roi%d" % (data_source, acq, roi))
            if not os.path.exists(outdir2):
                os.makedirs(outdir2)

            print(idx)
            print("Start saving Metadata for %s%03d" % (data_source, acq))
            hdf5storage.savemat(
                os.path.join(outdir2, "roi%02d_c%d" % (idx, P.c)),
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
                    'roi': roi,
                    'data_source': data_source,
                    'extent': extent,
                    "moniker": moniker,
                    'center_angle': center_angle,
                    'bimg_shape': bimg.shape
                },
            )
            print("End saving Metadata")

        # Display images via matplotlib
        outdir1 = os.path.join("scoring", "rois", image_type, "%s%03d_roi%d" % (data_source, acq, roi))
        if not os.path.exists(outdir1):
            os.makedirs(outdir1)
        plt.clf()
        plt.imshow(bimg, vmin=-40, cmap="gray", extent=extent, origin="upper")
        plt.suptitle("%s%03d (c = %d m/s)" % (data_source, acq, np.round(P.c)))
        plt.savefig(os.path.join(outdir1, "roi%02d.png" % (idx)))
        plt.show()

        # Save
        mdict = {
            "grid": grid,
            "data_source": data_source,
            "acq": acq,
            "extent": extent,
            "c": P.c}
        hdf5storage.savemat(os.path.join(outdir1, "roi%02d" % (idx)), mdict)

        speckle_brightness[idx] = np.mean(np.abs(iq))
        print("Sound speed %d, ave speckle brightness %f" % (P.c, speckle_brightness[idx]))


    outdir_speckle = os.path.join("speckle_brightness", "roi", "%s%03d" % (data_source, acq))
    if not os.path.exists(outdir_speckle):
        os.makedirs(outdir_speckle)

    print("Start saving speckle brightness data for %s%03d" % (data_source, acq))
    hdf5storage.savemat(
        os.path.join(outdir_speckle, "roi%02d_c%d-c%d_step%d" % (roi, sound_speed[0], sound_speed[-1], speed_step)),
        {
            "speckle_brightness": speckle_brightness,
            # "speckle_brightness_norm": speckle_brightness_norm,
            'f0': P.fc,
            'fs': P.fs,
            'sound_speed': sound_speed,
            'acq': acq,
            'data_source': data_source,
            "moniker": moniker,
            'center_angle': center_angle,
        },
    )
    print("End saving speckle brightness data")




if __name__ == "__main__":
    for i in range(6,7):
        rois_invivo(i)

