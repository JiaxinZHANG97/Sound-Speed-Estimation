%% Display DAS and SLSC B-mode images using RF data (plane wave)
% Author: Jiaxin Zhang (jzhan295@jhu.edu)
% Version 1: 03/20/2024

% Note:
% This script uses CUBDL IQ data saved from Python scripts. To
% access the Python scripts, download them from CUBDL GitLab:
% https://gitlab.com/dongwoon.hyun/cubdl/-/tree/master and ask Jiaxin for
% three other scripts: test_submissions_saveIQ.py, das_torch_saveIQ.py, and
% save_IQdata.py.
% Ask Jiaxin to access the IQtoRF and beamforming scripts:
% iq2rf_jz.m and beamformer_SLSC_PW_US_linear.m

%% Load delayed IQ data (plane wave) and metadata
load('')

metadata.Sys.f0 = f0 * 1e-6; % [MHz]
metadata.Sys.fs = fs * 1e-6; % [MHz]
metadata.Sys.N_pw = size(idelayed_pw,3);
metadata.Sys.pitch = 0.254 * 1e-3; % [m]
metadata.Sys.N_ele = size(idelayed_pw,2);
metadata.Sys.N_ch = bimg_shape{2};
metadata.Sys.extent = cell2mat(extent);
metadata.Sys.N_samples = size(idelayed_pw,1);
metadata.Sys.true_data = bimg_shape{1};

% idelay_pw1 = idelayed_pw(:,:,38);
% qdelay_pw1 = qdelayed_pw(:,:,38);

RF_delay_data = iq2rf_jz(idelayed_pw,qdelayed_pw,f0,fs,1,1,center_angle); % multiple plane waves

%% DAS
if center_angle
    RF_das = sum(RF_delay_data,2);
    RF_delay_data_cube = reshape(RF_delay_data, metadata.Sys.N_ch, metadata.Sys.true_data, metadata.Sys.N_ele); % single plane wave
    RF_delay_data_cube = permute(RF_delay_data_cube,[2,3,1]);
else
    RF_das = sum(sum(RF_delay_data,3),2);
    RF_delay_data_cube = reshape(RF_delay_data, metadata.Sys.N_ch, metadata.Sys.true_data, metadata.Sys.N_ele, metadata.Sys.N_pw); % multiple plane waves
    RF_delay_data_cube = permute(RF_delay_data_cube,[2,3,1,4]);
end

env = hilbert(RF_das/max(RF_das(:)));
bmode_db = db(abs(env));

bmode_db = reshape(bmode_db,metadata.Sys.N_ch,metadata.Sys.true_data);
bmode_db = bmode_db';

figure
imagesc(bmode_db,[-60 0])
colormap gray
axis image
colorbar
title([moniker,'-RF-DAS-',data_source,num2str(acq)],'Interpreter','none')
set(gca, 'FontSize', 16)
set(gcf,'Position',[600 200 700 700])

%% SLSC
metadata.US.SLSC.maxM = 50;
[slsc_rf,cc_rf,metadata,x_axis,z_axis] = beamformer_SLSC_PW_US_linear(RF_delay_data_cube,metadata);

lags = [1,2,3,4,5,10,15,20,25,30,35,40,45,50];
for l = 1:length(lags)
    lag = lags(l);
    slsc_rf_lag = slsc_rf(:,:,lag);
    slsc_dB_rf = db(slsc_rf_lag ./ max(slsc_rf_lag(:)));
    
    DR = 60;

    figure
    imagesc(x_axis,z_axis,slsc_dB_rf, [-DR, 0])
    axis image
    colorbar
    colormap gray
    title([moniker,'-RF-SLSC-M',int2str(lag),'-',data_source,num2str(acq)],'Interpreter','none')
    xlabel('Lateral (mm)');
    ylabel('Axial (mm)');
    set(gca, 'FontSize', 16)
    set(gcf,'Position',[600 200 700 700])
    saveas(gcf,['D:\Jiaxin\research\mLOC_CUBDL\results\EUT003\RF_',moniker,'_',data_source,num2str(acq),'_SLSC_M',num2str(lag),'.png'])
    saveas(gcf,['D:\Jiaxin\research\mLOC_CUBDL\results\EUT003\RF_',moniker,'_',data_source,num2str(acq),'_SLSC_M',num2str(lag),'.fig'])
    close
end











