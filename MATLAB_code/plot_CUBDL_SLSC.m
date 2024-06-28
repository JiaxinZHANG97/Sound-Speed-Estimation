%% plot CUBDL speckle brightness maximization results
epoch = 50;

figure
plot([0:epoch],cs)
xlabel('Iterations')
ylabel('Sound speed (m/s)')
set(gca, 'FontSize', 16)
set(findobj(gca, 'type', 'line'),'linew', 2);

figure
plot([0:epoch],losses)
xlabel('Iterations')
ylabel('Loss')
set(gca, 'FontSize', 16)
set(findobj(gca, 'type', 'line'),'linew', 2);

figure
plot([0:epoch],speckle_brightness_epoch)
xlabel('Iterations')
ylabel('Speckle brightness')
set(gca, 'FontSize', 16)
set(findobj(gca, 'type', 'line'),'linew', 2);


%% Plot speckle brightness vs. sound speed
figure
plot(sound_speed, speckle_brightness)
set(gca, 'FontSize', 16)
xlabel('Sound speed (m/s)')
ylabel('Speckle brightness')
set(findobj(gca, 'type', 'line'),'linew', 2);
xlim([c_range(1) c_range(2)])
xlim([1350 1700])
legend('ROI 1','ROI 2','ROI 3','ROI 4','ROI 5','Location','northeastoutside')

%% Plot FWHM vs. sound speed
figure
plot(sound_speed,x_fwhms)
xlabel('Sound speed (m/s)')
ylabel('Lateral FWHM (mm)')
set(gca, 'FontSize', 16)
set(findobj(gca, 'type', 'line'),'linew', 2);

figure
plot(sound_speed,z_fwhms)
xlabel('Sound speed (m/s)')
ylabel('Axial FWHM (mm)')
set(gca, 'FontSize', 16)
set(findobj(gca, 'type', 'line'),'linew', 2);

%% Plot contrast, CNR, gCNR vs. sound speed
% sound_speed = linspace(c_range(1),c_range(2),size(RF_max,1));

figure
plot(sound_speed,contrasts)
xlabel('Sound speed (m/s)')
ylabel('Contrast (dB)')
set(gca, 'FontSize', 16)
set(findobj(gca, 'type', 'line'),'linew', 2);
xlim([sound_speed(1) sound_speed(end)])

figure
plot(sound_speed,cnrs)
xlabel('Sound speed (m/s)')
ylabel('CNR')
set(gca, 'FontSize', 16)
set(findobj(gca, 'type', 'line'),'linew', 2);
xlim([sound_speed(1) sound_speed(end)])

figure
plot(sound_speed,gcnrs)
xlabel('Sound speed (m/s)')
ylabel('gCNR')
set(gca, 'FontSize', 16)
set(findobj(gca, 'type', 'line'),'linew', 2);
xlim([sound_speed(1) sound_speed(end)])

%% Plot 3D figure for sound speeds and M values
lags = 1:39; % 30% receive aperture: 0.3*128
num_lag = length(lags);
sound_speed = linspace(c_range(1),c_range(2),size(RF_max,1));

% [RF_max_max,I] = max(RF_max(:,1:num_lag));
% [RF_max_k_max,I] = max(RF_max_k(:,1:num_lag));
% [RF_max_kk_max,I] = max(RF_max_k(:,1:num_lag)k);
% [cc_max_max,I] = max(cc_max(:,1:num_lag));
[RF_min_min,I] = min(RF_min(:,1:num_lag));
% [RF_min_k_min,I] = min(RF_min_k(:,1:num_lag));
% [RF_min_kk_min,I] = min(RF_min_kk(:,1:num_lag));
% [cc_min_min,I] = min(cc_min(:,1:num_lag));
% [mu_roi_max,I] = max(mu_roi(:,1:num_lag));
% [mu_roi_min_min,I] = min(mu_roi_min(:,1:num_lag));
% [mu_all_min,I] = min(mu_all(:,1:num_lag));

I_speed = sound_speed(I);
speed_mode = mode(I_speed);
X_lag = repmat(lags,length(sound_speed),1);
Y_speed = repmat(sound_speed',1,length(lags));

disp(['Most frequent speed: ', num2str(speed_mode), ' m/s'])

figure
% surf(X_lag,Y_speed,RF_max(:,1:num_lag))
% surf(X_lag,Y_speed,RF_max_k(:,1:num_lag))
% surf(X_lag,Y_speed,RF_max_kk(:,1:num_lag))
% surf(X_lag,Y_speed,cc_max(:,1:num_lag))
surf(X_lag,Y_speed,RF_min(:,1:num_lag))
% surf(X_lag,Y_speed,RF_min_k(:,1:num_lag))
% surf(X_lag,Y_speed,RF_min_kk(:,1:num_lag))
% surf(X_lag,Y_speed,cc_min(:,1:num_lag))
% surf(X_lag,Y_speed,mu_roi(:,1:num_lag))
% surf(X_lag,Y_speed,mu_roi_min(:,1:num_lag))
% surf(X_lag,Y_speed,mu_all(:,1:num_lag))
hold on

% scatter3(lags,I_speed,RF_max_max,'filled','r')
% scatter3(lags,I_speed,RF_max_k_max,'filled','r')
% scatter3(lags,I_speed,RF_max_kk_max,'filled','r')
% scatter3(lags,I_speed,cc_max_max,'filled','r')
scatter3(lags,I_speed,RF_min_min,'filled','r')
% scatter3(lags,I_speed,RF_min_k_min,'filled','r')
% scatter3(lags,I_speed,RF_min_kk_min,'filled','r')
% scatter3(lags,I_speed,cc_min_min,'filled','r')
% scatter3(lags,I_speed,mu_roi_max,'filled','r')
% scatter3(lags,I_speed,mu_roi_min_min,'filled','r')
% scatter3(lags,I_speed,mu_all_min,'filled','r')

% xlabel('M')
xlabel('Cumulative lag')
ylabel('Sound speed (m/s)')
zlabel('Short-lag spatial coherence')
title(['Mode ', num2str(speed_mode), 'm/s'])
colorbar
set(gca, 'FontSize', 16)
xlim([lags(1) lags(end)])
ylim([sound_speed(1) sound_speed(end)])

% speed = sound_speed(I);






