%% Calculate Contrast-to-noise ratio

% INPUTS:
%   mu_t            -> (numeric) mean of a target region
%   mu_b            -> (numeric) mean of a backgroud region
%   sig_t           -> (numeric) standard deviation of a target region
%   sig_b           -> (numeric) standard deviation of a background region

% OUTPUT:
%   cnr             -> cnr measurement
function cnr = computeCnr(mu_t, mu_b, sig_t, sig_b)
	if(~isnumeric(mu_t) ...
			|| ~isnumeric(mu_b) ...
			|| ~isnumeric(sig_b) ...
			|| ~isnumeric(sig_t))
		error('All inputs must be numeric.');
	end

	cnr = abs(mu_t - mu_b) / sqrt((sig_t ^ 2) + (sig_b ^ 2));
end
