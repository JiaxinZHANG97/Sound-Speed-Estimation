%% Display DAS and SLSC B-mode images using IQ or RF data (plane wave)
% Author: Jiaxin Zhang (jzhan295@jhu.edu)
% Version 1: 03/20/2024
% Version 2: 06/24/2024

% Note:
% This script uses CUBDL IQ data saved from Python scripts. To
% access the Python scripts, download them from CUBDL GitLab:
% https://gitlab.com/dongwoon.hyun/cubdl/-/tree/master and ask Jiaxin for
% three other scripts: test_submissions_saveIQ.py, das_torch_saveIQ.py, and
% save_IQdata.py.
% Ask Jiaxin to access the IQtoRF and beamforming scripts:
% iq2rf_jz.m and beamformer_SLSC_PW_US_linear.m

%% Load not-delayed IQ data (plane wave) and metadata
metadata.Sys.c = c; % [m/s]
metadata.Sys.f0 = f0 * 1e-6; % [MHz]
metadata.Sys.fs = fs * 1e-6; % [MHz]
metadata.Sys.N_pw = size(idata,1);
% metadata.Sys.pitch = 0.3 * 1e-3; % [m]
metadata.Sys.N_ele = size(idata,2);
metadata.Sys.N_ch = bimg_shape{2};
metadata.Sys.extent = cell2mat(extent);
metadata.Sys.N_samples_orig = size(idata,3);
metadata.Sys.N_samples = size(rxapo,2);
metadata.Sys.true_data = bimg_shape{1};

data_source_full = sprintf('%s%03s-c%s', data_source,num2str(acq),num2str(metadata.Sys.c));
folder_result = ['D:\Jiaxin\research\mLOC_CUBDL\image_soundspeed\', data_source_full];
if ~exist(folder_result, 'dir')
   mkdir(folder_result)
end


% Initialize the output array
idelayed_pw = zeros(metadata.Sys.N_samples, metadata.Sys.N_ele, metadata.Sys.N_pw);
qdelayed_pw = zeros(size(idelayed_pw));
% idas = zeros(1,metadata.Sys.N_samples);
% qdas = zeros(1,metadata.Sys.N_samples);
% Loop over angles and elements
for t = 1:metadata.Sys.N_pw
    td = txdel(t,:);
    ta = txapo(t,:);
    for r = 1:metadata.Sys.N_ele
        rd = rxdel(r,:);
        ra = rxapo(r,:);

        delays = td + rd;
        dgs = (delays * 2 + 1) / size(idata,3) - 1;

        iq_i = idata(t,r,:);
        iq_q = qdata(t,r,:);
        iq_x = linspace(-1,1,metadata.Sys.N_samples_orig);
        ifoc = interp1(iq_x,iq_i(1,:), dgs,'linear',0);
        qfoc = interp1(iq_x,iq_q(1,:), dgs,'linear',0);

        % Apply apodization, reshape, and add to running sum
        apods = ta .* ra;
        idelayed_pw(:,r,t) = ifoc .* apods;
        qdelayed_pw(:,r,t) = qfoc .* apods;
        % idas = idas + ifoc .* apods;
        % qdas = qdas + qfoc .* apods;
    end
end
 

% Finally, restore the original pixel grid shape
idas = reshape(idas, bimg_shape{2},bimg_shape{1});
qdas = reshape(qdas, bimg_shape{2},bimg_shape{1});

iq_das = idas + 1i*qdas;
bimg = db(abs(iq_das));
bimg = bimg - max(bimg(:));
bimg = bimg';

x_axis = linspace(metadata.Sys.extent(1),metadata.Sys.extent(2),metadata.Sys.N_ch); % [mm]
z_axis = linspace(metadata.Sys.extent(4),metadata.Sys.extent(3),metadata.Sys.true_data); % [mm]

figure
imagesc(x_axis, z_axis, bimg, [-60 0])
xlabel('Lateral (mm)');
ylabel('Axial (mm)');
colormap gray
axis image
colorbar
title([moniker,'-IQ-DAS-',data_source_full],'Interpreter','none')
set(gca, 'FontSize', 16)
set(gcf,'Position',[600 200 700 700])
% saveas(gcf,[folder_result,'\RF_',moniker,'_',data_source_full,'_DAS_IQ.png'])
% saveas(gcf,[folder_result,'\RF_',moniker,'_',data_source_full,'_DAS_IQ.fig'])
%% Load delayed IQ data (plane wave) and metadata
metadata.Sys.c = c; % [m/s]
metadata.Sys.f0 = f0 * 1e-6; % [MHz]
metadata.Sys.fs = fs * 1e-6; % [MHz]
metadata.Sys.N_pw = size(idelayed_pw,3);
% metadata.Sys.pitch = 0.3 * 1e-3; % [m]
metadata.Sys.N_ele = size(idelayed_pw,2);
metadata.Sys.N_ch = bimg_shape{2};
metadata.Sys.extent = cell2mat(extent);
metadata.Sys.N_samples = size(idelayed_pw,1);
metadata.Sys.true_data = bimg_shape{1};

% data_source_full = sprintf('%s%03s', data_source,num2str(acq));
data_source_full = sprintf('%s%03s-c%s', data_source,num2str(acq),num2str(metadata.Sys.c));
folder_result = ['D:\Jiaxin\research\mLOC_CUBDL\image_soundspeed\', data_source_full];
if ~exist(folder_result, 'dir')
   mkdir(folder_result)
end

% ang_center = (metadata.Sys.N_pw+1)/2;
% idelayed_pw1 = idelayed_pw(:,:,ang_center);
% qdelayed_pw1 = qdelayed_pw(:,:,ang_center);

%% IQ to RF
% RF_delay_data = iq2rf_jz(idelayed_pw1,qdelayed_pw1,f0,fs,1,1,center_angle); % single plane waves
RF_delay_data = iq2rf_jz(idelayed_pw,qdelayed_pw,f0,fs,1,1,center_angle); % multiple plane waves

clear delays dgs idas idata ifoc iq_i iq_q iq_x qdas qdata qfoc ra rd rxapo rxdel ta td txapo txdel idelayed_pw qdelayed_pw

%% DAS
if center_angle
    RF_das = sum(RF_delay_data,2);
    RF_delay_data_cube = reshape(RF_delay_data, metadata.Sys.N_ch, metadata.Sys.true_data, metadata.Sys.N_ele); % single plane wave
    RF_delay_data_cube = permute(RF_delay_data_cube,[2,3,1]);
else
    RF_das = sum(sum(RF_delay_data,3),2);
    % Old code
    % RF_delay_data_cube = reshape(RF_delay_data, metadata.Sys.N_ch, metadata.Sys.true_data, metadata.Sys.N_ele, metadata.Sys.N_pw); % multiple plane waves
    % RF_delay_data_cube = permute(RF_delay_data_cube,[2,3,1,4]);
    % New code
    RF_delay_data_cube = zeros(metadata.Sys.true_data, metadata.Sys.N_ele, metadata.Sys.N_ch, metadata.Sys.N_pw, 'like', RF_delay_data);
    for pw = 1:metadata.Sys.N_pw
        temp_cube = reshape(RF_delay_data(:, :, pw), metadata.Sys.N_ch, metadata.Sys.true_data, metadata.Sys.N_ele);
        RF_delay_data_cube(:, :, :, pw) = permute(temp_cube, [2, 3, 1]);
    end
end

% env = hilbert(RF_das/max(RF_das(:)));
% bmode_db = db(abs(env));

env = abs(hilbert(RF_das));
env = reshape(env,metadata.Sys.N_ch,metadata.Sys.true_data);
env = env';
bmode_db = db(env/max(env(:)));

x_axis = linspace(metadata.Sys.extent(1),metadata.Sys.extent(2),metadata.Sys.N_ch); % [mm]
z_axis = linspace(metadata.Sys.extent(4),metadata.Sys.extent(3),metadata.Sys.true_data); % [mm]

save(fullfile(folder_result, 'env.mat'),'env')

figure
imagesc(x_axis, z_axis, bmode_db,[-60 0])
xlabel('Lateral (mm)');
ylabel('Axial (mm)');
colormap gray
axis image
colorbar
title([moniker,'-RF-DAS-',data_source_full],'Interpreter','none')
set(gca, 'FontSize', 16)
set(gcf,'Position',[600 200 700 700])
saveas(gcf,[folder_result,'\RF_',moniker,'_',data_source_full,'_DAS.png'])
saveas(gcf,[folder_result,'\RF_',moniker,'_',data_source_full,'_DAS.fig'])
    
%% SLSC
zero_out_flag = true;
metadata.US.SLSC.maxM = 40;
[slsc_rf,~,metadata,x_axis,z_axis] = beamformer_SLSC_PW_US_linear(RF_delay_data_cube,metadata,zero_out_flag);

lags = [1,2,3,4,5,10,13,15,20,25,30,35,40];%,45,50];

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
    title([moniker,'-RF-SLSC-M',int2str(lag),'-',data_source_full],'Interpreter','none')
    xlabel('Lateral (mm)');
    ylabel('Axial (mm)');
    set(gca, 'FontSize', 16)
    set(gcf,'Position',[600 200 700 700])
    saveas(gcf,[folder_result,'\RF_',moniker,'_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.png'])
    saveas(gcf,[folder_result,'\RF_',moniker,'_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.fig'])
    close
end


%% Draw rectangular ROI
ROI_rect = drawrectangle;

ROI_xlim = [ROI_rect.Position(1),ROI_rect.Position(1)+ROI_rect.Position(3)];
ROI_ylim = [ROI_rect.Position(2),ROI_rect.Position(2)+ROI_rect.Position(4)];
fprintf("ROI xlim = [%.2f, %.2f]mm\n",ROI_xlim(1),ROI_xlim(2));
fprintf("ROI ylim = [%.2f, %.2f]mm\n",ROI_ylim(1),ROI_ylim(2));

