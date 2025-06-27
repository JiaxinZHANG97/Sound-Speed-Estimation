%% Manually select rectangular ROIs and get ROI location information
% Author: Jiaxin Zhang (jzhan295@jhu.edu)
% Version 1: 10/04/2024

%% Make folders
data_source_name = sprintf('%s%03s', data_source,num2str(acq));
data_source_full = sprintf('%s%03s-c%s', data_source,num2str(acq),num2str(metadata.Sys.c));
folder_result = ['D:\Jiaxin\research\mLOC_CUBDL\image_soundspeed\', data_source_name, '\',data_source_full];
if ~exist(folder_result, 'dir')
   mkdir(folder_result)
end

%% Draw rectangular ROI
ROI_rect = drawrectangle();

% Add a listener for the 'MovingROI' event to capture the new position when moving
addlistener(ROI_rect, 'MovingROI', @(src, evt) updatePosition(src,0));
%%
ROI_xlim = [ROI_rect_0(1),ROI_rect_0(1)+ROI_rect_0(3)];
ROI_ylim = [ROI_rect_0(2),ROI_rect_0(2)+ROI_rect_0(4)];
ROI_size_halfX = ROI_rect_0(3)/2;
ROI_size_halfY = ROI_rect_0(4)/2;
ROI_xcenter = ROI_rect_0(1) + ROI_size_halfX;
ROI_ycenter = ROI_rect_0(2) + ROI_size_halfY;

fprintf("ROI xlim = [%.2f, %.2f] mm\n",ROI_xlim(1),ROI_xlim(2));
fprintf("ROI ylim = [%.2f, %.2f] mm\n",ROI_ylim(1),ROI_ylim(2));
fprintf("ROI xcenter = %.2f mm\n",ROI_xcenter);
fprintf("ROI ycenter = %.2f mm\n",ROI_ycenter);
fprintf("ROI size half x = %.2f mm\n",ROI_size_halfX);
fprintf("ROI size half y = %.2f mm\n\n",ROI_size_halfY);
%% background ROIs (divide the bg ROI into multiple equal ones)
roi_num = 2;
direction = "column";
for r = 1:roi_num
    if direction == "row"
        temp_roi = ROI_rect_0;
        temp_roi(4) = temp_roi(4)/roi_num;
    elseif direction == "column"
        temp_roi = ROI_rect_0;
        temp_roi(3) = temp_roi(3)/roi_num;
    end
    ROI_rect_copy = drawrectangle('Position', temp_roi);
    addlistener(ROI_rect_copy, 'MovingROI', @(src, evt) updatePosition(src,r));
end
%% print new ROI position
ROI_rect_new = ROI_rect_1;
ROI_xlim_new = [ROI_rect_new(1),ROI_rect_new(1)+ROI_rect_new(3)];
ROI_ylim_new = [ROI_rect_new(2),ROI_rect_new(2)+ROI_rect_new(4)];
ROI_size_halfX_new = ROI_rect_new(3)/2;
ROI_size_halfY_new = ROI_rect_new(4)/2;
ROI_xcenter_new = ROI_rect_new(1) + ROI_size_halfX_new;
ROI_ycenter_new = ROI_rect_new(2) + ROI_size_halfY_new;

fprintf("ROI xlim bg = [%.2f, %.2f] mm\n",ROI_xlim_new(1),ROI_xlim_new(2));
fprintf("ROI ylim bg = [%.2f, %.2f] mm\n",ROI_ylim_new(1),ROI_ylim_new(2));
fprintf("ROI xcenter bg = %.2f mm\n",ROI_xcenter_new);
fprintf("ROI ycenter bg = %.2f mm\n",ROI_ycenter_new);
fprintf("ROI size half x bg = %.2f mm\n",ROI_size_halfX_new);
fprintf("ROI size half y bg = %.2f mm\n\n",ROI_size_halfY_new);
%% histogram of ROI
roi = 5;
folder_result_histogram = fullfile(folder_result,['histogram-roi',num2str(roi)]);
if ~exist(folder_result_histogram, 'dir')
   mkdir(folder_result_histogram)
end
folder_result_h = fullfile(folder_result_histogram,'histogram_roi');
if ~exist(folder_result_h, 'dir')
   mkdir(folder_result_h)
end
folder_result_r = fullfile(folder_result_histogram,'rightEdge');
if ~exist(folder_result_r, 'dir')
   mkdir(folder_result_r)
end
folder_result_l = fullfile(folder_result_histogram,'leftEdge');
if ~exist(folder_result_l, 'dir')
   mkdir(folder_result_l)
end
folder_result_c = fullfile(folder_result_histogram,'centerBin');
if ~exist(folder_result_c, 'dir')
   mkdir(folder_result_c)
end


ROI_x = ROI_xlim;
ROI_y = ROI_ylim;

[~, idx_x1] = min(abs(x_axis-ROI_x(1)));
[~, idx_x2] = min(abs(x_axis-ROI_x(2)));
[~, idx_z1] = min(abs(z_axis-ROI_y(1)));
[~, idx_z2] = min(abs(z_axis-ROI_y(2)));

lags = 1:39;
for l = 1:length(lags)
    lag = lags(l);
    slsc_rf_lag = slsc_rf(:,:,lag); % no zero-out
    slsc_rf_lag_roi = slsc_rf_lag(idx_z1:idx_z2,idx_x1:idx_x2);
    slsc_rf_lag_roi_norm = (slsc_rf_lag_roi-min(slsc_rf_lag_roi(:)))/(max(slsc_rf_lag_roi(:))-min(slsc_rf_lag_roi(:)));
    
    
    figure
    imagesc(x_axis(idx_x1:idx_x2), z_axis(idx_z1:idx_z2), slsc_rf_lag_roi)
    % imagesc(slsc_rf_lag_roi_norm)
    title(['lag = ',num2str(lag), ' ROI'])
    colorbar
    xlabel('Lateral (mm)');
    ylabel('Axial (mm)');
    set(gca, 'FontSize', 16)
    axis image
    saveas(gcf,fullfile(folder_result_h,['roi_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.png']))
    saveas(gcf,fullfile(folder_result_h,['roi_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.fig']))
    close

    % histogram of ROI
    figure
    h=histogram(slsc_rf_lag_roi);
    title(['lag = ',num2str(lag), ' ROI'])
    xlabel('SLSC pixel value')
    ylabel('Count')
    set(gca, 'FontSize', 16)
    saveas(gcf,fullfile(folder_result_h,['histogram_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.png']))
    saveas(gcf,fullfile(folder_result_h,['histogram_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.fig']))
    binCounts = h.Values;
    [~, modeBinIdx] = max(binCounts);
    binEdges = h.BinEdges;
    leftEdge = binEdges(modeBinIdx);
    rightEdge = binEdges(modeBinIdx + 1);
    centerBin = (leftEdge + rightEdge)/2;
    close
    
    % mean of values below right edge
    slsc_img_rightEdge = slsc_rf_lag_roi;
    slsc_img_rightEdge(slsc_img_rightEdge>rightEdge)=max(slsc_img_rightEdge(:));
    mu_rightEdge = mean(slsc_img_rightEdge(:),'omitnan');
    figure
    imagesc(x_axis(idx_x1:idx_x2), z_axis(idx_z1:idx_z2), slsc_img_rightEdge)
    title(['lag = ',num2str(lag), ' Right Edge'])
    colorbar
    xlabel('Lateral (mm)');
    ylabel('Axial (mm)');
    set(gca, 'FontSize', 16)
    axis image
    saveas(gcf,fullfile(folder_result_r,['rightEdge_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.png']))
    saveas(gcf,fullfile(folder_result_r,['rightEdge_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.fig']))
    close
    
    % mean of values below left edge
    slsc_img_leftEdge = slsc_rf_lag_roi;
    slsc_img_leftEdge(slsc_img_leftEdge>leftEdge)=max(slsc_img_leftEdge(:));
    mu_leftEdge = mean(slsc_img_leftEdge(:),'omitnan');
    figure
    imagesc(x_axis(idx_x1:idx_x2), z_axis(idx_z1:idx_z2), slsc_img_leftEdge)
    title(['lag = ',num2str(lag), ' Left Edge'])
    colorbar
    xlabel('Lateral (mm)');
    ylabel('Axial (mm)');
    set(gca, 'FontSize', 16)
    axis image
    saveas(gcf,fullfile(folder_result_l,['leftEdge_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.png']))
    saveas(gcf,fullfile(folder_result_l,['leftEdge_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.fig']))
    close

    % mean of values below mode bin center
    slsc_img_centerBin = slsc_rf_lag_roi;
    slsc_img_centerBin(slsc_img_centerBin>centerBin)=max(slsc_img_centerBin(:));
    mu_centerBin = mean(slsc_img_centerBin(:),'omitnan');
    figure
    imagesc(x_axis(idx_x1:idx_x2), z_axis(idx_z1:idx_z2), slsc_img_centerBin)
    title(['lag = ',num2str(lag), ' Center Bin'])
    colorbar
    xlabel('Lateral (mm)');
    ylabel('Axial (mm)');
    set(gca, 'FontSize', 16)
    axis image
    saveas(gcf,fullfile(folder_result_c,['centerBin_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.png']))
    saveas(gcf,fullfile(folder_result_c,['centerBin_',data_source_full,'_SLSC_M',num2str(lag),'_DR',num2str(DR),'.fig']))
    close
end