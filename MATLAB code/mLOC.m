clear
d = uigetdir(pwd, 'Select a folder');
files = dir(fullfile(d, '*.mat'));
base_folder = files(1).folder;
idcs = strfind(base_folder,'\');
base_folder = base_folder(1:idcs(end)-1);

target_depth = 38; % Unit: mm. 0927:50; 1023: 25; 1102-flat: 28; 1102-curve(location1:38,location2:45,location3:50)
target_Xcenter = -46; % 0927(location1:-28,location2:-10,location3:+8);1023(location1:-19,location2:-9,location3:1);1102-flat(location1:-30,location2:-20,location3:-11,location4:-1); 1102-curve:(location1:-46,location2:-31,location3:-12)

R_recon = 91;

% sound_speed = linspace(1230,1850,63); % 1540 +- 20%
sound_speed = linspace(1080,2000,93); % 1540 +- 30%
% sound_speed = linspace(1350,2000,66);
% sound_speed = linspace(1710,2000,30);
maxM = 1;
lags = [1];%,maxM];


RF_max = zeros(length(files),length(lags),length(sound_speed));
z_center_true_mm = zeros(length(files),length(lags),length(sound_speed));
x_center_true_mm = zeros(length(files),length(lags),length(sound_speed));
z_fwhm_true = zeros(length(files),length(lags),length(sound_speed));
x_fwhm_true = zeros(length(files),length(lags),length(sound_speed));
contrast = zeros(length(files),length(lags),length(sound_speed));
cnr = zeros(length(files),length(lags),length(sound_speed));
snr = zeros(length(files),length(lags),length(sound_speed));
gcnr = zeros(length(files),length(lags),length(sound_speed));

%%
% For each .mat file, run through all wavelen=gth
for file_ind=1:length(files)
    file_path = fullfile(files(file_ind).folder,files(file_ind).name);
    file_name = files(file_ind).name;
    disp(file_ind)
    %% PA
    % base_folder = 'D:\Jiaxin\PULSE\PA\verasonics\data\';
    % file_path = 'C:\Users\jiaxi\Documents\Jiaxin\PULSE\PA\verasonics\data\flexible_array\20220331_184247.mat'; %
    % file_path = '/Volumes/DORIS/verasonics/data/flexible_array/20220331_184247.mat';
    % file_path = '/Volumes/DORIS/verasonics/data/flexible_array/20220331_184549.mat';
    %     file_path = 'C:\Users\jiaxi\Documents\Jiaxin\PULSE\PA\verasonics\data\flexible_0331_med_50mm\fiberposition_new\20220331_194158.mat';
    % file_path = 'C:\Users\jiaxi\Documents\Jiaxin\PULSE\PA\verasonics\data\flexible_0415_large_40mm\fiberposition_new\20220415_184423.mat';
    % file_path = append(base_folder, 'flexible_0415_large_40mm\20220415_183626.mat'); % large fiber position 8 mm
    % file_path = append(base_folder, 'flexible_0331_med_50mm\20220331_175456.mat'); % medium fiber position 7 mm
    % file_path = append(base_folder, 'flexible_0428_small_40mm\20220428_192020.mat'); % small fiber position 7 mm
    % file_path = append(base_folder, 'flexible_array\20230927\20230927_124559.mat'); % test
    %     file_path = 'C:\Users\jiaxi\Documents\Jiaxin\PULSE\PA\verasonics\data\flexible_0428_small_40mm\fiberposition\20220428_192020.mat';
    %     file_path = 'C:\Users\jiaxi\Documents\Jiaxin\PULSE\PA\verasonics\data\flexible_array\20220504_132507.mat'; % registration
    %     file_path = 'C:\Users\jiaxi\Documents\Jiaxin\PULSE\PA\verasonics\data\flexible_array\20220810_175355.mat'; % multiple
    %     file_path = 'C:\Users\jiaxi\Documents\Jiaxin\PULSE\PA\verasonics\data\flexible_array\20221122_225409.mat'; % liver
    %     file_path = 'C:\Users\jiaxi\Documents\Jiaxin\PULSE\PA\verasonics\data\flexible_0428_small_40mm\20220428_194755';%20220428_174417';%20220428_194755';%20220428_194347';%20220428_194548';
    %     [PA_raw_data, PA_dB, metadata, x_ax, z_ax] = beamformer_DAS_PA_flexible_verasonics_v2(file_path,'radius_DAS',81.3,'radius_DSC',81.3,'shape','concave','scale',1);
   
    
    % RF_max{file_ind} = zeros(length(lags),length(sound_speed));
    % z_center_true_mm{file_ind} = zeros(length(lags),length(sound_speed));
    % x_center_true_mm{file_ind} = zeros(length(lags),length(sound_speed));
    % z_fwhm_true{file_ind} = zeros(length(lags),length(sound_speed));
    % x_fwhm_true{file_ind} = zeros(length(lags),length(sound_speed));
    % contrast{file_ind} = zeros(length(lags),length(sound_speed));
    % cnr{file_ind} = zeros(length(lags),length(sound_speed));
    % snr{file_ind} = zeros(length(lags),length(sound_speed));
    % gcnr{file_ind} = zeros(length(lags),length(sound_speed));
    
     
    for c = 1:length(sound_speed)
        disp(['Sound speed recon = ', num2str(sound_speed(c))])
        R_DAS = R_recon;
        R_DSC = R_recon;
        R_DAS_t = strrep(num2str(R_DAS),'.','');
        
        s_speed = sound_speed(c);
        
    
        [delay_data, metadata, PA_raw_data, RxMux] = delay_PA_flexible_experiment_fast(file_path,'radius_DAS',R_DAS,'sound_speed',s_speed,'scale',1);
        % [delay_data, metadata, PA_raw_data, RxMux] = delay_PA_flexible_experiment_delayMTX(file_path,delay_mtx,'radius_DAS',R_DAS,'sound_speed',s_speed,'scale',1);
        % [delay_data,metadata,RxMux] = delay_PA_verasonics_linear(file_path,'sound_speed',s_speed);


        % metadata.SysPA.DAS.index_window = 2; % 0 -> rectwin; 2 -> hamming
        % [PA_img, RF_sum, metadata, x_ax, z_ax] = beamformer_DAS_PA_flexible_experiment(delay_data,metadata,RxMux,'radius_DSC',R_DSC,'shape','flat');
        % % 
        % PA_dB = db(PA_img ./ max(PA_img(:)));
        % x_ax_center = x_ax-x_ax(end)/2;
        
        metadata.SysPA.SLSC.maxM = maxM;
        [PA_slsc_img, cc, metadata, x_ax, z_ax] = beamformer_SLSC_PA_flexible_experiment(delay_data,metadata,RxMux,'radius_DSC',R_DSC,'shape','flat');
        % metadata.PA.SLSC.maxM = maxM;
        % [PA_slsc_img, cc, metadata, x_ax, z_ax] = beamformer_SLSC_PA_linear(delay_data,metadata,RxMux);

    %     % interpolation for no scan conversion
    %     x_ax_interp = linspace(0, metadata.SysPA.img_width*1e3, metadata.SysPA.img_x); % interpolation for more samples, finer grid
    %     z_ax_interp = linspace(0, metadata.SysPA.img_dep*1e3, metadata.SysPA.img_z); % interpolation for more samples, finer grid
    % 
    %     X = repmat(x_ax,length(z_ax),1);
    %     Z = repmat(z_ax',1,length(x_ax));
    %     Xq = repmat(x_ax_interp,metadata.SysPA.img_z,1);
    %     Zq = repmat(z_ax_interp',1,metadata.SysPA.img_x);
        %% Plot scan line
    
    
        %% DAS
        % % % Plot DAS
        % figure
        % imagesc(x_ax_center, z_ax, PA_dB, [-15 0]);
        % axis image; colorbar;
        % colormap hot;
        % set(gca, 'FontSize', 16)
        % xlabel('Lateral (mm)');
        % ylabel('Axial (mm)');
        % title(['Flexible array DAS', newline, 'R = ', num2str(R_DAS),' mm C = ', num2str(s_speed)]);
        % 
        % % Plot DAS combine
        % figure
        % imagesc(x_ax_center, z_ax, PA_dB, [-15 0]);
        % axis image; colorbar;
        % colormap hot;
        % set(gca, 'FontSize', 16)
        % xlabel('Lateral (mm)');
        % ylabel('Axial (mm)');
        % title(['Flexible array DAS combine', newline, 'R = ', num2str(R_DAS),' mm C = ', num2str(s_speed)]);
    %% SLSC
        % Plot SLSC PA image at optimal lag M
        for l = 1:length(lags)
            
            lag = lags(l);
    %         lag = metadata.SysPA.SLSC.maxM;
            slsc_PA = squeeze(PA_slsc_img(:,:,lag));
            PA_slsc_dB = db(slsc_PA ./ max(slsc_PA(:)));
    
            PA_img = slsc_PA;
            PA_dB = PA_slsc_dB;
            x_ax_center = x_ax-x_ax(end)/2;
    
    %         PA_img = interp2(X,Z,PA_img, Xq, Zq);
    %         PA_dB = interp2(X,Z,PA_dB, Xq, Zq);
    %         x_ax = x_ax_interp;
    %         z_ax = z_ax_interp;
    %         x_ax_center = x_ax-x_ax(end)/2;
            % 
            % figure
            % imagesc(x_ax_center, z_ax, PA_dB, [-15,0]);
            % axis image; colorbar;
            % colormap hot;
            % set(gca, 'FontSize', 16)
            % xlabel('Lateral (mm)');
            % ylabel('Axial (mm)');
            % title(['SLSC Large (center 64ele) Lag', num2str(lag), newline, 'R = ', num2str(R_DAS),' mm C = ', num2str(s_speed)]);
    %         saveas(gcf,[base_folder,'flexible_0415_large_40mm\position_8mm\slsc\wrongC_R71_lag1lag6\183626_center64_lag',num2str(lag),'_R',num2str(R_DAS_t),'_c',num2str(s_speed),'.png'])
    %         saveas(gcf,[base_folder,'flexible_0415_large_40mm\position_8mm\slsc\wrongC_R71_lag1lag6\183626_center64_lag',num2str(lag),'_R',num2str(R_DAS_t),'_c',num2str(s_speed),'.fig'])
    %         close
    
    
            %% FWHM
            %     mm2samp = 1/metadata.SysPA.wls2mm*metadata.SysPA.wls2samp*2;
            [z_orig, x_orig] = size(PA_dB);
    
            mm2samp = length(z_ax)/(z_ax(end));
            samp2mm = (z_ax(end))/length(z_ax);
            mm2sampX = length(x_ax)/(x_ax(end));
            samp2mmX = (x_ax(end))/length(x_ax);
            % target_Xcenter = x_ax(end)/2+8; % location1: -28, location2: -10, location3: +8
            roi_z_center = round(target_depth*mm2samp); 
            roi_x_center = round((x_ax(end)/2+target_Xcenter)*mm2sampX);%round(x_ax(end)/2*mm2samp);
            half_z_range = round(15*mm2samp); % region for finding the max brightness 
            half_x_range = round(15*mm2sampX); % region for finding the max brightness
        %     z_range = [roi_z_center-half_z_range:1:roi_z_center+half_z_range];
        %     x_range = [roi_x_center-half_x_range:1:roi_x_center+half_x_range];
    
            z_start = roi_z_center-half_z_range;
            z_end = roi_z_center+half_z_range;
            x_start = roi_x_center-half_x_range;
            x_end = roi_x_center+half_x_range;
            if z_end>z_orig
                z_end=z_orig;
            end
            if x_end>x_orig
                x_end=x_orig;
            end
            if z_start<=0
                z_start=1;
            end
            if x_start<=0
                x_start=1;
            end
    
            z_range = z_start:z_end;
            x_range = x_start:x_end;
    
            mask_roi = NaN(size(PA_dB));
            mask_roi(z_range,x_range) = 1;
            mask_roi_b = circshift(mask_roi, 303,2);
    
            img_roi = PA_dB.*mask_roi;
            img_roi_b = PA_dB.*mask_roi_b;
            RF_roi = PA_img .* mask_roi;
    
        %     img_roi_slsc = PA_slsc_dB.*mask_roi;
        %     img_roi_b_slsc = PA_slsc_dB.*mask_roi_b;
        %     RF_roi_slsc = PA_slsc_img .* mask_roi;
    
            roi = img_roi(z_range,x_range);
        %     roi_slsc = img_roi_slsc(z_range,x_range);
    
            x_range_roi = x_range/mm2sampX;
            z_range_roi = z_range/mm2samp;
    
        % %     % ROI - rough
        %     figure
        %     imagesc(x_ax, z_ax, img_roi);
        %     title(['Curve Surface'])
        %     xlabel('Lateral Distance (mm)'), ylabel('Axial (mm)')
        %     colorbar
        %     axis image
        % 
    %         figure
    %         imagesc(x_range_roi, z_range_roi, roi, [-15,0]);
    %         title(['3'])
    %         xlabel('Lateral (mm)'), ylabel('Axial (mm)')
    %         colorbar
    %         axis image
    %         set(gca, 'FontSize', 16)
    %         title(['Simulation ROI', newline, 'R = ', num2str(R_DAS), ' mm']);
    %         colormap hot;
    
    
    
            %% find max brightness (PA_img or RF_env)
            [RF_max(file_ind,l,c),I] = max(RF_roi(:)); % single img
            [roi_z_center_true, roi_x_center_true] = ind2sub(size(RF_roi),I);
            z_center_true_mm(file_ind,l,c) = roi_z_center_true/mm2samp;
            x_center_true_mm(file_ind,l,c) = roi_x_center_true/mm2sampX;
            half_z_range_roi = round(5*mm2samp); % 10*10 ROI region with target at center 
            half_x_range_roi = round(5*mm2sampX); % 10*10 ROI region with target at center
            z_start = roi_z_center_true-half_z_range_roi;
            z_end = roi_z_center_true+half_z_range_roi;
            x_start = roi_x_center_true-half_x_range_roi;
            x_end = roi_x_center_true+half_x_range_roi;
            if z_end>z_orig
                z_end=z_orig;
            end
            if x_end>x_orig
                x_end=x_orig;
            end
            if z_start<=0
                z_start=1;
            end
            if x_start<=0
                x_start=1;
            end
            z_range_true = [z_start:1:z_end];
            x_range_true = [x_start:1:x_end];
    
            mask_roi_true = NaN(size(PA_dB));
            mask_roi_true(z_range_true,x_range_true) = 1;
            mask_roi_b_true = circshift(mask_roi_true, 303, 2);
    
            img_roi_true = PA_dB.*mask_roi_true;
            img_roi_b_true = PA_dB.*mask_roi_b_true;
            RF_roi_true = PA_img .* mask_roi_true;
    
            roi_true = img_roi_true(z_range_true,x_range_true);
            x_range_roi_true = x_range_true/mm2sampX;
            z_range_roi_true = z_range_true/mm2samp;
        % 
            %     % ROI - true
            % figure
            % imagesc(x_ax, z_ax, img_roi_true);
            % title(['Curve Surface'])
            % xlabel('Lateral Distance (mm)'), ylabel('Axial (mm)')
            % colorbar
            % set(gca, 'FontSize', 16)
            % axis image

    %         figure
    %         imagesc(x_range_roi_true, z_range_roi_true, roi_true, [-40,0]);
    %         xlabel('Lateral (mm)'), ylabel('Axial (mm)')
    %         colorbar
    %         axis image
    %         set(gca, 'FontSize', 16)
    %         title(['Simulation ROI true', newline, 'R = ', num2str(R_DAS), ' mm']);
    %         colormap hot;
        % %     saveas(gcf,[saveDir,'ROI_DAS',R_DAS_t,'.png'])
        % %     saveas(gcf,[saveDir,'ROI_DAS',R_DAS_t,'.fig'])
        % %     close
    
            %% resolution (FWHM)
            z_roi = PA_dB(z_range,roi_x_center);
    
            z_ind = find(z_roi>=max(z_roi)-6);
            z_fw = z_ind(end)-z_ind(1);
            %     z_fwhm(file_ind) = z_fw/metadata.SysPA.wls2samp*metadata.SysPA.wls2mm; % for loop
            % z_fwhm = z_fw/metadata.SysPA.wls2samp*metadata.SysPA.wls2mm; % single img
            z_fwhm = z_fw*samp2mm; % single img
    
            x_roi = PA_dB(roi_z_center, x_range);
            %     x_roi = img_roi(roi_z_center,:);
            x_ind = find(x_roi>=max(x_roi)-6);
            x_fw = x_ind(end)-x_ind(1);
            %     z_fwhm_true(file_ind) = z_fw/metadata.SysPA.wls2samp*metadata.SysPA.wls2mm; % for loop
            % x_fwhm = x_fw/metadata.SysPA.wls2samp*metadata.SysPA.wls2mm;
            x_fwhm = x_fw*samp2mmX;
    
    
            %     x_signal = img_roi(roi_z_center,:);
            %     x_signal_interp = interp1(0:x_orig-1,x_signal,x_ax);
            %     x_ax_ind = find(x_signal_interp);
            %     x = linspace(0,x_ax(end),x_orig);
            %     x_roi_orig = PA_dB(roi_z_center,:);
            %     x_roi_interp = (interp1(x,x_roi_orig,x_ax));
            %     x_roi = x_roi_interp(x_ax_ind(1):x_ax_ind(end));
            % 
            %     x_ind = find(x_roi>=max(x_roi)-6);
            % %     x_fwhm(file_ind) = x_ax(x_ind(end))-x_ax(x_ind(1)); % for loop
            %     x_fwhm = x_ax(x_ind(end))-x_ax(x_ind(1)); % single img
    
            %% resolution (FWHM) true
            z_roi = PA_dB(z_range_true,roi_x_center_true);
    
            z_ind = find(z_roi>=max(z_roi)-6);
            z_fw = z_ind(end)-z_ind(1);
            %     z_fwhm_true(file_ind) = z_fw/metadata.SysPA.wls2samp*metadata.SysPA.wls2mm; % for loop
            % z_fwhm_true = z_fw/metadata.SysPA.wls2samp*metadata.SysPA.wls2mm;
            z_fwhm_true(file_ind,l,c) = z_fw*samp2mm; % single img
    
            x_roi = PA_dB(roi_z_center_true, x_range_true);
            x_ind = find(x_roi>=max(x_roi)-6);
            x_fw = x_ind(end)-x_ind(1);
            %     z_fwhm_true(file_ind) = z_fw/metadata.SysPA.wls2samp*metadata.SysPA.wls2mm; % for loop
            % x_fwhm_true = x_fw/metadata.SysPA.wls2samp*metadata.SysPA.wls2mm;
            x_fwhm_true(file_ind,l,c) = x_fw*samp2mmX; % single img
    
    
            %% contrast
            half_z_range_p = round(0.3*mm2samp); 
            half_x_range_p = round(0.3*mm2sampX);
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
    
            roi_shift = round(20*mm2sampX);%-280;
            mask_roi_true_p = zeros(size(PA_dB));
            mask_roi_true_p(z_range_true_p,x_range_true_p) = 1;
            mask_roi_b_true_p = circshift(mask_roi_true_p, roi_shift, 2);
    
            img_roi_true_p = PA_dB.*mask_roi_true_p;
            img_roi_b_true_p = PA_dB.*mask_roi_b_true_p;
            RF_roi_true_p = PA_img .* mask_roi_true_p;
    
            % % ROI - point
            % figure
            % imagesc(x_ax, z_ax, img_roi_true_p);
            % title(['Curve Surface'])
            % xlabel('Lateral Distance (mm)'), ylabel('Axial (mm)')
            % colorbar
            % axis image
            % 
            %     % ROI - background
            % figure
            % imagesc(x_ax, z_ax, img_roi_b_true_p);
            % title(['Curve Surface'])
            % xlabel('Lateral Distance (mm)'), ylabel('Axial (mm)')
            % colorbar
            % axis image
    
            img_roi_RF_p = PA_img.*mask_roi_true_p;
            sum_t = sum(img_roi_RF_p(:),'omitnan');
            area_t = sum(mask_roi_true_p(:),'omitnan');
            mu_t = sum_t/area_t;
    
            mask_b_p = mask_roi_b_true_p;
            img_b_RF_p = PA_img.*mask_b_p;
            sum_b = sum(img_b_RF_p(:),'omitnan');
            area_b = sum(mask_b_p(:),'omitnan');
            mu_b = sum_b/area_b;
    
            contrast(file_ind,l,c) = computeContrast(mu_t, mu_b, 'db', true);
    
            %% CNR
            img_roi_RF_p = PA_img.*mask_roi_true_p;
            sum_t = sum(img_roi_RF_p(:),'omitnan');
            area_t = sum(mask_roi_true_p(:),'omitnan');
            mu_t = sum_t/area_t;
            std_t = std(img_roi_RF_p(:),'omitnan');
    
            mask_b_p = mask_roi_b_true_p;
            img_b_RF_p = PA_img.*mask_b_p;
            sum_b = sum(img_b_RF_p(:),'omitnan');
            area_b = sum(mask_b_p(:),'omitnan');
            mu_b = sum_b/area_b;
            std_b = std(img_b_RF_p(:),'omitnan');
    
            cnr(file_ind,l,c) = computeCnr(mu_t, mu_b, std_t, std_b);
    
            %% SNR
            img_roi_RF_p = PA_img.*mask_roi_true_p;
            sum_t = sum(img_roi_RF_p(:),'omitnan');
            area_t = sum(mask_roi_true_p(:),'omitnan');
            mu_t = sum_t/area_t;
    
            mask_b_p = mask_roi_b_true_p;
            img_b_RF_p = PA_img.*mask_b_p;
            std_b = std(img_b_RF_p(:),'omitnan');
    
            snr(file_ind,l,c) = computeSnr(mu_t, std_b, 'db', true);
    
            %% gCNR
            img_roi_RF_p = PA_img.*mask_roi_true_p;
            target_roi = PA_img(z_range_true_p, x_range_true_p);
            sum_t = sum(img_roi_RF_p(:),'omitnan');
            area_t = sum(mask_roi_true_p(:),'omitnan');
            % mu_t = sum_t/area_t;
            % std_t = std(img_roi_RF_p(:));
    
            mask_b_p = mask_roi_b_true_p;
            img_b_RF_p = PA_img.*mask_b_p;
            bg_roi = PA_img(z_range_true_p, x_range_true_p+ roi_shift);
            sum_b = sum(img_b_RF_p(:),'omitnan');
            area_b = sum(mask_b_p(:),'omitnan');
            % mu_b = sum_b/area_b;
            % std_b = std(img_b_RF_p(:));
    
            gcnr(file_ind,l,c) = computeGcnr(target_roi, bg_roi);
            
            
    %         save(fullfile(base_folder, append('flexible_0415_large_40mm\results_position_8mm\slsc\wrongR\RF_max_lag',num2str(lag),'.mat')),'RF_max')
    %         save(fullfile(base_folder, append('flexible_0415_large_40mm\results_position_8mm\slsc\wrongR\z_center_true_mm_lag',num2str(lag),'.mat')),'z_center_true_mm')
    %         save(fullfile(base_folder, append('flexible_0415_large_40mm\results_position_8mm\slsc\wrongR\x_center_true_mm_lag',num2str(lag),'.mat')),'x_center_true_mm')
    %         save(fullfile(base_folder, append('flexible_0415_large_40mm\results_position_8mm\slsc\wrongR\z_fwhm_true_lag',num2str(lag),'.mat')),'z_fwhm_true')
    %         save(fullfile(base_folder, append('flexible_0415_large_40mm\results_position_8mm\slsc\wrongR\x_fwhm_true_lag',num2str(lag),'.mat')),'x_fwhm_true')
    %         save(fullfile(base_folder, append('flexible_0415_large_40mm\results_position_8mm\slsc\wrongR\contrast_lag',num2str(lag),'.mat')),'contrast')
    %         save(fullfile(base_folder, append('flexible_0415_large_40mm\results_position_8mm\slsc\wrongR\cnr_lag',num2str(lag),'.mat')),'cnr')
    %         save(fullfile(base_folder, append('flexible_0415_large_40mm\results_position_8mm\slsc\wrongR\snr_lag',num2str(lag),'.mat')),'snr')
    %         save(fullfile(base_folder, append('flexible_0415_large_40mm\results_position_8mm\slsc\wrongR\gcnr_lag',num2str(lag),'.mat')),'gcnr')
            
        end
            
        
    end
end
%%
save(fullfile(base_folder, 'results\noDSC_wrongC_R91_lag1_center64_actualF\RF_max_all.mat'),'RF_max')
save(fullfile(base_folder, 'results\noDSC_wrongC_R91_lag1_center64_actualF\z_center_true_mm_all.mat'),'z_center_true_mm')
save(fullfile(base_folder, 'results\noDSC_wrongC_R91_lag1_center64_actualF\x_center_true_mm_all.mat'),'x_center_true_mm')
save(fullfile(base_folder, 'results\noDSC_wrongC_R91_lag1_center64_actualF\z_fwhm_true_all.mat'),'z_fwhm_true')
save(fullfile(base_folder, 'results\noDSC_wrongC_R91_lag1_center64_actualF\x_fwhm_true_all.mat'),'x_fwhm_true')
save(fullfile(base_folder, 'results\noDSC_wrongC_R91_lag1_center64_actualF\contrast_all.mat'),'contrast')
save(fullfile(base_folder, 'results\noDSC_wrongC_R91_lag1_center64_actualF\cnr_all.mat'),'cnr')
save(fullfile(base_folder, 'results\noDSC_wrongC_R91_lag1_center64_actualF\snr_all.mat'),'snr')
save(fullfile(base_folder, 'results\noDSC_wrongC_R91_lag1_center64_actualF\gcn_all.mat'),'gcnr')

