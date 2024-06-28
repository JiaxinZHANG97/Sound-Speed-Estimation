%% mLOC metric
% Author: Jiaxin Zhang (jzhan295@jhu.edu)
% Version 1: November 2023 (original name:
% image_quality_analysis_slsc_wrongC_loop.m) This version was used for
% sound speed estimation using flexible array and PA
% Version 2: 03/25/2024 This version was modified for mLOC and US (CUBDL)
% using delayed channel datasaved from Python scripts
% Version 3: 04/07/2024 Calculates max or min SLSC within ROI using
% non-delayed channel data saved from Python scripts

clear
d = uigetdir(pwd, 'Select a folder');
files = dir(fullfile(d, '*.mat')); % #files = #speeds
base_folder = files(1).folder;
idcs = strfind(base_folder,'\');
base_folder = base_folder(1:idcs(end)-1);

maxM = 64;
lags = 1:maxM;

k = 5;
kk = 10;

cc_max = zeros(length(files),length(lags));
RF_max = zeros(length(files),length(lags));
RF_max_k = zeros(length(files),length(lags));
RF_max_kk = zeros(length(files),length(lags));
mu_roi = zeros(length(files),length(lags));
mu_all = zeros(length(files),length(lags));
z_center_true_mm = zeros(length(files),length(lags));
x_center_true_mm = zeros(length(files),length(lags));
cc_min = zeros(length(files),length(lags));
RF_min = zeros(length(files),length(lags));
RF_min_k = zeros(length(files),length(lags));
RF_min_kk = zeros(length(files),length(lags));
mu_roi_min = zeros(length(files),length(lags));
z_center_true_mm_min = zeros(length(files),length(lags));
x_center_true_mm_min = zeros(length(files),length(lags));

%%
% Each .mat file corresponds to one sound speed
for file_ind=1:length(files)
    file_path = fullfile(files(file_ind).folder,files(file_ind).name);
    file_name = files(file_ind).name;
    disp(file_ind)

    load(file_path)

    if file_ind == 1
        c_range(1) = c;
    elseif file_ind == length(files)
        c_range(2) = c;
    end
    if c==1540
        extent_1540 = cell2mat(extent);
    end

    %% Load delayed IQ data (plane wave) and metadata
    % metadata.Sys.f0 = f0 * 1e-6; % [MHz]
    % metadata.Sys.fs = fs * 1e-6; % [MHz]
    % metadata.Sys.c = c; % [m/s]
    % metadata.Sys.N_pw = size(idelayed_pw,3);
    % metadata.Sys.N_ele = size(idelayed_pw,2);
    % metadata.Sys.N_ch = bimg_shape{2};
    % metadata.Sys.extent = cell2mat(extent);
    % metadata.Sys.N_samples = size(idelayed_pw,1);
    % metadata.Sys.true_data = bimg_shape{1};

    %% Load non-delayed IQ data (plane wave) and metadata
    metadata.Sys.c = c; % [m/s]
    metadata.Sys.f0 = f0 * 1e-6; % [MHz]
    metadata.Sys.fs = fs * 1e-6; % [MHz]
    metadata.Sys.N_pw = size(idata,1);
    metadata.Sys.N_ele = size(idata,2);
    metadata.Sys.N_ch = bimg_shape{2};
    metadata.Sys.extent = cell2mat(extent);
    metadata.Sys.N_samples_orig = size(idata,3);
    metadata.Sys.N_samples = size(rxapo,2);
    metadata.Sys.true_data = bimg_shape{1};
    
    % data_source_full = sprintf('%s%03s-c%s', data_source,num2str(acq),num2str(metadata.Sys.c));
    % folder_result = ['D:\Jiaxin\research\mLOC_CUBDL\image_soundspeed\', data_source_full];
    % if ~exist(folder_result, 'dir')
    %    mkdir(folder_result)
    % end
    
    % Initialize the output array
    idelayed_pw = zeros(metadata.Sys.N_samples, metadata.Sys.N_ele, metadata.Sys.N_pw);
    qdelayed_pw = zeros(size(idelayed_pw));
    % Loop over angles and elements
    for t = 1:metadata.Sys.N_pw
        td = txdel(t,:);
        ta = txapo(t,:);
        for r = 1:metadata.Sys.N_ele
            rd = rxdel(r,:);
            ra = rxapo(r,:);
            % Convert delays to be used with interp2
            delays = td + rd;
            dgs = (delays * 2 + 1) / size(idata,3) - 1;
            % Interpolate using interp2 and vectorize using reshape
            iq_i = idata(t,r,:);
            iq_q = qdata(t,r,:);
            iq_x = linspace(-1,1,metadata.Sys.N_samples_orig);
            ifoc = interp1(iq_x,iq_i(1,:), dgs,'linear',0);
            qfoc = interp1(iq_x,iq_q(1,:), dgs,'linear',0);

            % Apply apodization, reshape, and add to running sum
            apods = ta .* ra;
            idelayed_pw(:,r,t) = ifoc .* apods;
            qdelayed_pw(:,r,t) = qfoc .* apods;
        end
    end
    %%
    disp(['Sound speed recon = ', num2str(c)])

    data_source_full = sprintf('%s%03s_roi%s', data_source,num2str(acq),num2str(roi));
    folder_base = ['D:\Jiaxin\research\mLOC_CUBDL\multiLOC\roi\', data_source_full];
    if ~exist(folder_base, 'dir')
       mkdir(folder_base)
    end

    folder_result = ['D:\Jiaxin\research\mLOC_CUBDL\multiLOC\roi\', data_source_full,'\results'];
    if ~exist(folder_result, 'dir')
       mkdir(folder_result)
    end
    
    RF_delay_data = iq2rf_jz(idelayed_pw,qdelayed_pw,f0,fs,1,1,center_angle); % multiple plane waves

    if center_angle
        RF_delay_data_cube = reshape(RF_delay_data, metadata.Sys.N_ch, metadata.Sys.true_data, metadata.Sys.N_ele); % single plane wave
        RF_delay_data_cube = permute(RF_delay_data_cube,[2,3,1]);
    else
        % RF_delay_data_cube = reshape(RF_delay_data, metadata.Sys.N_ch, metadata.Sys.true_data, metadata.Sys.N_ele, metadata.Sys.N_pw); % multiple plane waves
        % RF_delay_data_cube = permute(RF_delay_data_cube,[2,3,1,4]);
        RF_delay_data_cube = zeros(metadata.Sys.true_data, metadata.Sys.N_ele, metadata.Sys.N_ch, metadata.Sys.N_pw, 'like', RF_delay_data);
        for pw = 1:metadata.Sys.N_pw
            temp_cube = reshape(RF_delay_data(:, :, pw), metadata.Sys.N_ch, metadata.Sys.true_data, metadata.Sys.N_ele);
            RF_delay_data_cube(:, :, :, pw) = permute(temp_cube, [2, 3, 1]);
        end
    end

    zero_out_flag = false; % Do not zero out negative values when looking for optimal SLSC in ROI
    % zero_out_flag = true;
    % metadata.US.SLSC.k_factor = 1.56;
    metadata.US.SLSC.maxM = maxM;
    [slsc_rf,cc_rf,metadata,x_axis,z_axis] = beamformer_SLSC_PW_US_linear(RF_delay_data_cube,metadata,zero_out_flag);

    samp2mm = abs(z_axis(2) - z_axis(1));
    mm2samp = 1/samp2mm;
    samp2mmX = abs(x_axis(2) - x_axis(1));
    mm2sampX = 1/samp2mmX;
    %% SLSC
    % Calculate max/min SLSC at each M
    for l = 1:length(lags)
        
        lag = lags(l);
        cc_lag = squeeze(cc_rf(:,lag,:));
        slsc_lag = squeeze(slsc_rf(:,:,lag));
        slsc_dB = db(slsc_lag ./ max(slsc_lag(:)));

        slsc_img = slsc_lag;
        DR = 40;
        
        % % % 
        % figure
        % imagesc(x_axis,z_axis,slsc_dB, [-DR, 0])
        % % imagesc(x_axis,z_axis,slsc_lag)%, [-DR, 0])
        % axis image
        % colorbar
        % colormap gray
        % title([moniker,'-RF-SLSC-M',int2str(lag),'-c',num2str(c),'-',data_source_full],'Interpreter','none')
        % xlabel('Lateral (mm)');
        % ylabel('Axial (mm)');
        % set(gca, 'FontSize', 16)
        % set(gcf,'Position',[600 200 700 700])
        % saveas(gcf,[folder_base,'\RF_',moniker,'_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'_c',num2str(c),'.png'])
        % saveas(gcf,[folder_base,'\RF_',moniker,'_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'_c',num2str(c),'.fig'])
        % close


        %% find max brightness (PA_img or RF_env)
        [z_orig, x_orig] = size(slsc_img);
        [cc_max(file_ind,l),~] = max(cc_lag(:)); % single img
        [RF_max(file_ind,l),I] = max(slsc_img(:)); % single img
        [roi_z_center_true, roi_x_center_true] = ind2sub(size(slsc_img),I);
        z_center_true_mm(file_ind,l) = z_axis(1) + roi_z_center_true/mm2samp;
        x_center_true_mm(file_ind,l) = x_axis(1) + roi_x_center_true/mm2sampX;
        
        % half_z_range_roi = round(1*mm2samp); % 6*6 ROI region with target at center 
        % half_x_range_roi = round(1*mm2sampX); % 6*6 ROI region with target at center
        % z_start = roi_z_center_true-half_z_range_roi;
        % z_end = roi_z_center_true+half_z_range_roi;
        % x_start = roi_x_center_true-half_x_range_roi;
        % x_end = roi_x_center_true+half_x_range_roi;
        % 
        % [z_orig, x_orig] = size(slsc_img);
        % if z_end>z_orig
        %     z_end=z_orig;
        % end
        % if x_end>x_orig
        %     x_end=x_orig;
        % end
        % if z_start<=0
        %     z_start=1;
        % end
        % if x_start<=0
        %     x_start=1;
        % end
        % z_range_true = [z_start:1:z_end];
        % x_range_true = [x_start:1:x_end];
        % 
        % mask_roi_true = NaN(size(slsc_img));
        % mask_roi_true(z_range_true,x_range_true) = 1;
        % 
        % RF_roi_true = slsc_img .* mask_roi_true;

        % Top k values
        [top_k,I_k]=maxk(slsc_img(:),k); % adjust ROI: RF_roi_true, fixed ROI : slsc_img
        RF_max_k(file_ind,l) = mean(top_k);
        [roi_z_center_true_k, roi_x_center_true_k] = ind2sub(size(slsc_img),I_k); % adjust ROI: RF_roi_true, fixed ROI : slsc_img

        [top_kk,I_kk]=maxk(slsc_img(:),kk); % adjust ROI: RF_roi_true, fixed ROI : slsc_img
        RF_max_kk(file_ind,l) = mean(top_kk);
        [roi_z_center_true_kk, roi_x_center_true_kk] = ind2sub(size(slsc_img),I_kk); % adjust ROI: RF_roi_true, fixed ROI : slsc_img

        % 0.5mm*0.5mm ROI
        half_z_range_p = round(0.25*mm2samp); 
        half_x_range_p = round(0.25*mm2sampX);
        z_start_p = roi_z_center_true-half_z_range_p;
        z_end_p = roi_z_center_true+half_z_range_p;
        x_start_p = roi_x_center_true-half_x_range_p;
        x_end_p = roi_x_center_true+half_x_range_p;
        if z_end_p>z_orig
            z_end_p=z_orig;
        end
        if x_end_p>x_orig
            x_end_p=x_orig;
        end
        if z_start_p<=0
            z_start_p=1;
        end
        if x_start_p<=0
            x_start_p=1;
        end
        z_range_true_p = [z_start_p:1:z_end_p];
        x_range_true_p = [x_start_p:1:x_end_p];

        mask_roi_true_p = NaN(size(slsc_img));
        mask_roi_true_p(z_range_true_p,x_range_true_p) = 1;

        img_roi_RF_p = slsc_img.*mask_roi_true_p;
        sum_t = sum(img_roi_RF_p(:),'omitnan');
        area_t = sum(mask_roi_true_p(:),'omitnan');
        mu_roi(file_ind,l) = sum_t/area_t;


        %% find min brightness (PA_img or RF_env)
        [z_orig, x_orig] = size(slsc_img);
        [cc_min(file_ind,l),~] = min(cc_lag(:));
        [RF_min(file_ind,l),I] = min(slsc_img(:)); % single img
        [roi_z_center_true, roi_x_center_true] = ind2sub(size(slsc_img),I);
        z_center_true_mm_min(file_ind,l) = z_axis(1) + roi_z_center_true/mm2samp;
        x_center_true_mm_min(file_ind,l) = x_axis(1) + roi_x_center_true/mm2sampX;
        
        % half_z_range_roi = round(3*mm2samp); % 6*6 ROI region with target at center 
        % half_x_range_roi = round(3*mm2sampX); % 6*16 ROI region with target at center
        % z_start = roi_z_center_true-half_z_range_roi;
        % z_end = roi_z_center_true+half_z_range_roi;
        % x_start = roi_x_center_true-half_x_range_roi;
        % x_end = roi_x_center_true+half_x_range_roi;
        % 
        % [z_orig, x_orig] = size(slsc_img);
        % if z_end>z_orig
        %     z_end=z_orig;
        % end
        % if x_end>x_orig
        %     x_end=x_orig;
        % end
        % if z_start<=0
        %     z_start=1;
        % end
        % if x_start<=0
        %     x_start=1;
        % end
        % z_range_true = [z_start:1:z_end];
        % x_range_true = [x_start:1:x_end];
        % 
        % mask_roi_true = NaN(size(slsc_img));
        % mask_roi_true(z_range_true,x_range_true) = 1;
        % 
        % RF_roi_true = slsc_img .* mask_roi_true;

        % Top k values
        [min_k,I_k]=mink(slsc_img(:),k); % adjust ROI: RF_roi_true, fixed ROI : slsc_img
        RF_min_k(file_ind,l) = mean(min_k);
        [roi_z_center_true_k, roi_x_center_true_k] = ind2sub(size(slsc_img),I_k); % adjust ROI: RF_roi_true, fixed ROI : slsc_img


        [min_kk,I_kk]=mink(slsc_img(:),kk); % adjust ROI: RF_roi_true, fixed ROI : slsc_img
        RF_min_kk(file_ind,l) = mean(min_kk);
        [roi_z_center_true_kk, roi_x_center_true_kk] = ind2sub(size(slsc_img),I_kk); % adjust ROI: RF_roi_true, fixed ROI : slsc_img

        % 0.5mm*0.5mm ROI
        half_z_range_p = round(0.25*mm2samp); 
        half_x_range_p = round(0.25*mm2sampX);
        z_start_p = roi_z_center_true-half_z_range_p;
        z_end_p = roi_z_center_true+half_z_range_p;
        x_start_p = roi_x_center_true-half_x_range_p;
        x_end_p = roi_x_center_true+half_x_range_p;
        if z_end_p>z_orig
            z_end_p=z_orig;
        end
        if x_end_p>x_orig
            x_end_p=x_orig;
        end
        if z_start_p<=0
            z_start_p=1;
        end
        if x_start_p<=0
            x_start_p=1;
        end
        z_range_true_p = [z_start_p:1:z_end_p];
        x_range_true_p = [x_start_p:1:x_end_p];

        mask_roi_true_p = NaN(size(slsc_img));
        mask_roi_true_p(z_range_true_p,x_range_true_p) = 1;

        img_roi_RF_p = slsc_img.*mask_roi_true_p;
        sum_t = sum(img_roi_RF_p(:),'omitnan');
        area_t = sum(mask_roi_true_p(:),'omitnan');
        mu_roi_min(file_ind,l) = sum_t/area_t;

        %% Average of whole ROI
        mu_all(file_ind,l) = mean(slsc_img(:));
      
    end
        
end
%%
save(fullfile(folder_result, 'cc_max.mat'),'cc_max')
save(fullfile(folder_result, 'RF_max.mat'),'RF_max')
save(fullfile(folder_result, 'RF_max_k.mat'),'RF_max_k')
save(fullfile(folder_result, 'RF_max_kk.mat'),'RF_max_kk')
save(fullfile(folder_result, 'mu_roi.mat'),'mu_roi')
save(fullfile(folder_result, 'z_center_true_mm.mat'),'z_center_true_mm')
save(fullfile(folder_result, 'x_center_true_mm.mat'),'x_center_true_mm')

save(fullfile(folder_result, 'cc_min.mat'),'cc_min')
save(fullfile(folder_result, 'RF_min.mat'),'RF_min')
save(fullfile(folder_result, 'RF_min_k.mat'),'RF_min_k')
save(fullfile(folder_result, 'RF_min_kk.mat'),'RF_min_kk')
save(fullfile(folder_result, 'mu_roi_min.mat'),'mu_roi_min')
save(fullfile(folder_result, 'z_center_true_mm_min.mat'),'z_center_true_mm_min')
save(fullfile(folder_result, 'x_center_true_mm_min.mat'),'x_center_true_mm_min')

save(fullfile(folder_result, 'mu_all.mat'),'mu_all')

save(fullfile(folder_result, 'c_range.mat'),'c_range')
save(fullfile(folder_result, 'extent_1540.mat'),'extent_1540')

