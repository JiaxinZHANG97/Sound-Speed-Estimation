%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generates SLSC images using the compiled mex files

% Inputs:
%   delay_data:                 [samples x channels x elements x planewaves]
%   metadata:                   structure of imaging parameters     
%   metadata.SLSC.k_factor:     (optional) wavelength factor for kernel size
%   metadata.SLSC.maxM:         (optional) number of total lags

% Outputs:
%   slsc:                       SLSC image matrix [samples x channles x lags]
%   cc:                         coherence coefficient matrix
%   metadata.SLSC.x_axis:       lateral vector for plotting [mm]
%   metadata.SLSC.z_axis:       axial vector for plotting [mm]
%   metadata.SLSC.k_size:       axial size of the SLSC kernel 

% Example:
% [slsc,cc,metadata,x_axis,z_axis] = beamformer_SLSC_US_linear(delay_data,metadata)

% Modified from PULSE Lab code beamformer_SLSC_US_linear.m
% Version1: Jiaxin Zhang (jzhan295@jhu.edu)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [slsc,cc,metadata,x_axis,z_axis] = beamformer_SLSC_PW_US_linear(delay_data,metadata,zero_out_flag)
%% Loading SLSC parameters
if ~isfield(metadata,'US')
    metadata.US.SLSC.k_factor=1; 
    metadata.US.SLSC.maxM=50;
elseif ~isfield(metadata.US,'SLSC')
    metadata.US.SLSC.k_factor=1; 
    metadata.US.SLSC.maxM=50;
else
    if ~isfield(metadata.US.SLSC,'k_factor')
        metadata.US.SLSC.k_factor=1;
    end
    if ~isfield(metadata.US.SLSC,'maxM')
        metadata.US.SLSC.maxM=50;
    end   
end
    
k = metadata.US.SLSC.k_factor*metadata.Sys.fs/metadata.Sys.f0;
k = round(k)+1-mod(round(k),2); % Round to odd
metadata.US.SLSC.k_size=k;
maxM=metadata.US.SLSC.maxM;
cc = zeros(size(delay_data,1),maxM,size(delay_data,3));
slsc = zeros(size(delay_data,1),size(delay_data,3),maxM);

%% Modified for multiple plane wave
% cc_pw = zeros(size(delay_data,1),maxM,size(delay_data,3),metadata.Sys.N_pw);
for pw = 1:metadata.Sys.N_pw
    fprintf(['Plane wave: ',num2str(pw),'\n'])
    %% Remove DC along the axial dimension
    temp_data = delay_data(:,:,:,pw);
    for i = 1:size(delay_data,2)
        for j =  1:size(delay_data,3)
            temp_data(:,i,j) = delay_data(:,i,j,pw)-mean(delay_data(:,i,j,pw));
        end
    end

    %% Generate correlation coefficient matrix
    cc_temp = zeros(size(delay_data,1),maxM,size(delay_data,3));
    for ch = 1:metadata.Sys.N_ch
        temp = temp_data(:,:,ch);
        lag = makelagmat(1,size(temp,2),maxM);
        cc2 = slsc_mex(temp,lag,k,1);
        cc2(isnan(cc2)) = 0;
        cc_temp(:,:,ch) = cc2;
        % cc_pw(:,:,ch,pw) = cc2;
    end
    cc = cc + cc_temp;

    %% Sum along lags
    slsc_temp = zeros(size(delay_data,1),size(delay_data,3),maxM);
    for M=1:maxM
        slsc_temp(:,:,M) = squeeze(sum(cc_temp(:,1:M,:),2))/M;
    end
    if zero_out_flag
        slsc_temp(slsc_temp<0) = 0; % set negative SLSC image pixels to zero (Nair AA: Robust Short-Lag Spatial Coherence Imaging)
    end
    slsc = slsc + slsc_temp;
end
cc = cc / metadata.Sys.N_pw;
slsc = slsc / metadata.Sys.N_pw;

%% Axis generation
x_axis = linspace(metadata.Sys.extent(1),metadata.Sys.extent(2),metadata.Sys.N_ch); % [mm]
z_axis = linspace(metadata.Sys.extent(4),metadata.Sys.extent(3),metadata.Sys.true_data); % [mm]

metadata.US.x_axis=x_axis;
metadata.US.z_axis=z_axis;
end

