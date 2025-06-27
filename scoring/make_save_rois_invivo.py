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
import multiprocessing
import time

device = torch.device("cuda:0")


def rois_invivo(idx, outdir=os.path.join("scoring", "rois", "invivo")):
    user_input = input("test or save: ")
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
        # # roi3 (lesion)
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
        # roi = 13
        # xlims = [-2.5e-3, 1.5e-3]
        # zlims = [10e-3, 14e-3]
        # # roi14 (lesion)
        # roi = 14
        # xlims = [-2.3e-3, 3.3e-3]
        # zlims = [11.6e-3, 13.8e-3]
        # # roi15 (lesion)
        # roi = 15
        # xlims = [-2e-3, 1.5e-3]
        # zlims = [10.5e-3, 13.3e-3]
        # # roi16 (lesion)
        # roi = 16
        # xlims = [-0.7e-3, 3e-3]
        # zlims = [10.7e-3, 13.1e-3]
        # # roi17 (lesion)
        roi = 17
        xlims = [-2.5e-3, 1e-3]
        zlims = [11e-3, 13.9e-3]


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
        # roi = 3
        # xlims = [-6e-3, -1e-3]
        # zlims = [35.5e-3, 40.5e-3]
        # ROI 4
        roi = 4
        xlims = [0.16e-3, 5.18e-3]
        zlims = [17.47e-3, 21.46e-3]

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
        # # roi 3
        # roi = 3
        # xlims = [7e-3, 14.7e-3]
        # zlims = [39.5e-3, 46.5e-3]
        # # roi 4
        # roi = 4
        # xlims = [8.1e-3, 14.83e-3]
        # zlims = [40.12e-3, 46.49e-3]
        # # roi 5
        roi = 5
        xlims = [7.65e-3, 16.29e-3]
        zlims = [39.98e-3, 46.55e-3]

    elif idx == 4:
        data_source, acq = "JHU", 27
        P, xl, zl = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1
        # roi = 1
        # xlims = [-0.2e-3, 2.5e-3]
        # zlims = [14.4e-3, 16e-3]
        # # roi 2
        # roi = 2
        # ctr_x_select = 12.39e-3  # ROI center X selected by radiologist1 (at 1540 m/s)
        # ctr_z_select = 14.10e-3  # ROI center X selected by radiologist1 (at 1540 m/s)
        # roi_size_halfX = 1.03e-3
        # roi_size_halfZ = 0.63e-3
        # # roi 3
        # roi = 3
        # ctr_x_select = 12.3e-3  # ROI center X selected by radiologist2 (at 1540 m/s)
        # ctr_z_select = 14.4e-3  # ROI center X selected by radiologist2 (at 1540 m/s)
        # roi_size_halfX = 1.5e-3
        # roi_size_halfZ = 1e-3
        # # roi 4
        roi = 4
        ctr_x_select = 11.90e-3  # ROI center X selected by radiologist3 (at 1540 m/s)
        ctr_z_select = 14.42e-3  # ROI center Z selected by radiologist3 (at 1540 m/s)
        roi_size_halfX = 0.94e-3
        roi_size_halfZ = 0.84e-3

        width = xl[1] - xl[0]
        ctr_x_roi = width - ctr_x_select # P36 mass 1 is laterally flipped
        ctr_x = xl[0] + ctr_x_roi
        xlims = [ctr_x - roi_size_halfX, ctr_x + roi_size_halfX]
        zlims = [ctr_z_select - roi_size_halfZ, ctr_z_select + roi_size_halfZ]

    elif idx == 5:
        data_source, acq = "JHU", 24
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1
        roi = 1
        xlims = [-5.89e-3, -0.44e-3]
        zlims = [9.60e-3, 12.26e-3]
    elif idx == 6:
        data_source, acq = "JHU", 26
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1
        # roi = 1
        # xlims = [-2e-3, 1.5e-3]
        # zlims = [13.5e-3, 16.5e-3]
        # # roi 2
        # roi = 2
        # xlims = [-2.4e-3, 1.6e-3]
        # zlims = [13.5e-3, 16.5e-3]
        # # roi 3
        roi = 3
        xlims = [-3.3e-3, 2e-3]
        zlims = [13.5e-3, 16.6e-3]
    elif idx == 7:
        data_source, acq = "MYO", 1
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # roi 1
        # roi = 1
        # xlims = [12.5e-3, 17.5e-3]
        # zlims = [41.5e-3, 45.5e-3]
        # # roi 2
        roi = 2
        xlims = [14.15e-3, 16.97e-3]
        zlims = [14.71e-3, 17.18e-3]
    elif idx == 8:
        data_source, acq = "INS", 14
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # roi 1 (vessel)
        # roi = 1
        # xlims = [-1.5e-3, 1.5e-3]
        # zlims = [24.1e-3, 26.9e-3]
        # # roi 2 (vessel)
        # roi = 2
        # xlims = [-2e-3, 4e-3]
        # zlims = [21.6e-3, 27.3e-3]
        # # roi 3 (vessel - matches optimal contrast - but image not good)
        # roi = 3
        # xlims = [-10e-3, 8e-3]
        # zlims = [21.6e-3, 27.3e-3]
        # # roi 4 (point - good)
        # roi = 4
        # xlims = [7.1e-3, 10.7e-3]
        # zlims = [33.3e-3, 35.3e-3]
        # # roi 5 (vessel)
        # roi = 5
        # xlims = [-6.5e-3, -3.5e-3]
        # zlims = [23e-3, 26e-3]
        # # roi 6 (vessel)
        roi = 6
        xlims = [-5.45e-3, 0.32e-3]
        zlims = [22.64e-3, 26.35e-3]
    elif idx == 9:
        data_source, acq = "JHU", 29
        P, xl, zl = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1
        # roi = 1
        # xlims = [3.1e-3, 8e-3]
        # zlims = [12e-3, 14.3e-3]
        # # roi 2 (converge centerbin:1480 m/s)
        # roi = 2
        # xlims = [4e-3, 7.5e-3]
        # zlims = [12.5e-3, 13.7e-3]
        # # roi 3
        # roi = 3
        # xlims = [-1.2e-3, 2.5e-3]
        # zlims = [11.5e-3, 13.5e-3]
        # # roi 4
        # roi = 4
        # xlims = [3.3e-3, 8e-3]
        # zlims = [12.3e-3, 14.2e-3]
        # # roi 5 (converge centerbin:1515m/s)
        # roi = 5
        # xlims = [3.1e-3, 5.6e-3]
        # zlims = [12.5e-3, 13.9e-3]
        # # roi 6
        roi = 6
        ctr_x_select = 13e-3  # ROI center X selected by radiologist (at 1540 m/s)
        ctr_z_select = 7e-3  # ROI center Z selected by radiologist (at 1540 m/s)
        roi_size_halfX = 2.3e-3
        roi_size_halfZ = 2.5e-3

        ctr_x = xl[0] + ctr_x_select
        xlims = [ctr_x - roi_size_halfX, ctr_x + roi_size_halfX]
        zlims = [ctr_z_select - roi_size_halfZ, ctr_z_select + roi_size_halfZ]

    elif idx == 10:
        data_source, acq = "JHU", 30
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1
        # roi = 1
        # xlims = [-1.2e-3, 3.05e-3]
        # zlims = [11.42e-3, 13.77e-3]
        # # roi 2
        roi = 2
        xlims = [-1.76e-3, 2.90e-3]
        zlims = [11.72e-3, 14.59e-3]
    elif idx == 11:
        data_source, acq = "JHU", 31
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1
        # roi = 1
        # xlims = [0.02e-3, 5.61e-3]
        # zlims = [14.63e-3, 16.99e-3]
        # # roi 2
        # roi = 2
        # xlims = [0.02e-3, 3.82e-3]
        # zlims = [15.05e-3, 16.94e-3]
        # # roi 3
        # roi = 3
        # xlims = [4.74e-3, 9.35e-3]
        # zlims = [14.89e-3, 17.4e-3]
        # # roi 4
        # roi = 4
        # xlims = [4.53e-3, 8.53e-3]
        # zlims = [8.69e-3, 10.99e-3]
        # # roi 5
        # roi = 5
        # xlims = [7.35e-3, 11.3e-3]
        # zlims = [14.17e-3, 16.43e-3]
        # # roi 6
        # roi = 6
        # xlims = [3e-3, 8.28e-3]
        # zlims = [7.77e-3, 11.15e-3]
        # # # roi 7
        # roi = 7
        # xlims = [2.23e-3, 5.41e-3]
        # zlims = [9.97e-3, 12.53e-3]
        # # roi 8
        roi = 8
        xlims = [8.33e-3, 12.22e-3]
        zlims = [11.4e-3, 13.76e-3]
    elif idx == 12:
        data_source, acq = "JHU", 33
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1 (good results, but blank image at small SoS)
        # roi = 1
        # xlims = [-1.66e-3, 3.58e-3]
        # zlims = [12.71e-3, 16.39e-3]
        # # roi 2
        roi = 2
        xlims = [-1.32e-3, 5.71e-3]
        zlims = [13.05e-3, 16.28e-3]
    elif idx == 13:
        data_source, acq = "JHU", 34
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1
        # roi = 1
        # xlims = [2.34e-3, 10.57e-3]
        # zlims = [12.06e-3, 15.44e-3]
        # # roi 2
        # roi = 2
        # xlims = [6.96e-3, 12.57e-3]
        # zlims = [8.21e-3, 11.9e-3]
        # # roi 3
        roi = 3
        xlims = [-2.65e-3, 9.88e-3]
        zlims = [11.67e-3, 15.05e-3]
    elif idx == 14:
        data_source, acq = "JHU", 32
        P, _, _ = load_data(data_source, acq)
        image_type = "invivo"
        # # roi 1
        # roi = 1
        # xlims = [4.78e-3, 9.55e-3]
        # zlims = [5.98e-3, 9.21e-3]
        # # roi 2
        # roi = 2
        # xlims = [3.34e-3, 7.55e-3]
        # zlims = [9.27e-3, 10.96e-3]
        # # roi 3
        roi = 3
        xlims = [-1.43e-3, 2.06e-3]
        zlims = [13.37e-3, 15.84e-3]
    elif idx == 15:
        data_source, acq = "INS", 6
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # roi 1
        roi = 1
        xlims = [-14.75e-3, -7.56e-3]
        zlims = [35.02e-3, 41.22e-3]
        # # roi 2
        # roi = 2
        # xlims = [-1.91e-3, 5.28e-3]
        # zlims = [35.10e-3, 41.29e-3]
    elif idx == 16:
        data_source, acq = "INS", 26
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # roi 1
        # roi = 1
        # xlims = [-4.65e-3, -1.44e-3]
        # zlims = [29.27e-3, 30.77e-3]
        # # roi 2
        roi = 2
        xlims = [-1.20e-3, 2.11e-3]
        zlims = [29.42e-3, 31.75e-3]
    elif idx == 17:
        data_source, acq = "EUT", 6
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        # roi = 1
        # xlims = [0.72e-3, 5.96e-3]
        # zlims = [23.28e-3, 26.16e-3]
        # # ROI 2
        # roi = 2
        # xlims = [0.88e-3, 6.49e-3]
        # zlims = [24.2e-3, 27.03e-3]
        # # ROI 3
        roi = 3
        xlims = [1.92e-3, 7.08e-3]
        zlims = [19.34e-3, 21.62e-3]
    elif idx == 18:
        data_source, acq = "INS", 9
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        roi = 1
        xlims = [-6.52e-3, -0.89e-3]
        zlims = [40.49e-3, 43.58e-3]
    elif idx == 19:
        data_source, acq = "MYO", 4
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        roi = 1
        xlims = [10.90e-3, 13.86e-3]
        zlims = [23.52e-3, 26.50e-3]
    elif idx == 20:
        data_source, acq = "OSL", 10
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        # roi = 1
        # xlims = [6.11e-3, 9.09e-3]
        # zlims = [13.26e-3, 15.22e-3]
        # # ROI 2
        # roi = 2
        # xlims = [-12.73e-3, -3.14e-3]
        # zlims = [19.47e-3, 28.82e-3]
        # # ROI 3
        roi = 3
        xlims = [-14.47e-3, -8.61e-3]
        zlims = [33.26e-3, 40.86e-3]
    elif idx == 21:
        data_source, acq = "OSL", 7
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        # roi = 1
        # xlims = [-11.82e-3, -4.94e-3]
        # zlims = [35.56e-3, 42.82e-3]
        # # ROI 2
        roi = 2
        xlims = [-11.48e-3, -5.44e-3]
        zlims = [35.98e-3, 42.40e-3]
    elif idx == 22:
        data_source, acq = "OSL", 3
        P, _, _ = load_data(data_source, acq)
        image_type = "phantom"
        # # ROI 1
        roi = 1
        xlims = [-5.67e-3, 0.73e-3]
        zlims = [18.12e-3, 21.99e-3]
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

    for idx in range(nspeed): # range(17,21): range(nspeed):
        P.c = sound_speed[idx]
        wvln = P.c / P.fc
        dx = wvln / 3
        dz = dx  # Use square pixels
        c_ratio = (P.c / 1540)#.item()
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

        if user_input == "save":
            # Make 75-angle image
            if saveDelayed:
                das = DAS_PW_saveIQ(P, grid, rxfnum=fnum)
                idas, qdas, idelayed_pw, qdelayed_pw = das(x)
                # idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
                idelayed_pw, qdelayed_pw = idelayed_pw.detach().cpu().numpy(), qdelayed_pw.detach().cpu().numpy()

                # iq = idas + 1j * qdas
                # bimg = 20 * np.log10(np.abs(iq))  # Log-compress
                # bimg -= np.amax(bimg)  # Normalize by max value

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
                _, _, txapo, rxapo, txdel, rxdel = das(x)
                # idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
                txapo, rxapo = txapo.detach().cpu().numpy(), rxapo.detach().cpu().numpy()
                txdel, rxdel = txdel.detach().cpu().numpy(), rxdel.detach().cpu().numpy()

                # iq = idas + 1j * qdas
                # bimg = 20 * np.log10(np.abs(iq))  # Log-compress
                # bimg -= np.amax(bimg)  # Normalize by max value


                ### Save not delayed IQ data ###
                moniker = "ground_truth"
                center_angle = False
                outdir2 = os.path.join("Metadata", moniker, "roi", image_type, "%s%03d" % (data_source, acq), "%s%03d_roi%d" % (data_source, acq, roi))
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
                        'bimg_shape': grid.shape[:-1]#bimg.shape
                    },
                )
                print("End saving Metadata")

                # start_time = time.time()
                # num_workers = 2#multiprocessing.cpu_count()
                # params_list = [(outdir2, idx, P, txapo, rxapo, txdel, rxdel, acq, roi, data_source, extent, moniker, center_angle, bimg.shape)]
                # with multiprocessing.Pool(processes=num_workers) as pool:
                #     # Use pool.map to apply the save_data function to each set of parameters
                #     pool.map(save_Metadata, params_list)
                # end_time = time.time()
                # elapsed_time = end_time - start_time
                # print(f"Time used: {elapsed_time:.4f} seconds")

        if user_input == "test":
            das = DAS_PW(P, grid, rxfnum=fnum)
            idas, qdas = das(x)
            idas, qdas = idas.detach().cpu().numpy(), qdas.detach().cpu().numpy()
            iq = idas + 1j * qdas
            bimg = 20 * np.log10(np.abs(iq))  # Log-compress
            bimg -= np.amax(bimg)  # Normalize by max value

            plt.clf()
            plt.imshow(bimg, vmin=-40, cmap="gray", extent=extent, origin="upper")
            plt.suptitle("%s%03d (c = %d m/s)" % (data_source, acq, np.round(P.c)))
            # plt.savefig(os.path.join(outdir1, "roi%02d.png" % (idx)))
            plt.show()

def save_Metadata(params):
    outdir2, idx, P, txapo, rxapo, txdel, rxdel, acq, roi, data_source, extent, moniker, center_angle, bimg_shape = params
    print(f"Start saving Metadata for roi {idx}")
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
            'bimg_shape': bimg_shape
        },
    )
    print(f"End saving Metadata for roi {idx}")


if __name__ == "__main__":
    for i in range(9,10):
        rois_invivo(i)

