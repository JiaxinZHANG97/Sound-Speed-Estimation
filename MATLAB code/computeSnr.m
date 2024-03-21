%% Calculate Signal-to-noise ratio

% INPUTS:
%   mu_t            -> (numeric) mean of a target region
%   sig_b           -> (numeric) standard deviation of a background region

% INPUTS (optional):
%   db              -> (logical) compute the SNR in dB if true or the
%                      raw SNR if false
%   type           -> (string) compute amplitude of power SNR

% OUTPUT:
%   snr             -> (dB) SNR measurement
function snr = computeSnr(mu_t, sig_b, varargin)
	p = inputParser();

	if(~isnumeric(mu_t) || ~isnumeric(sig_b))
		error('Inputs mu_t and sig_b must be numeric.');
	end

	addParameter(p, 'db', true, @islogical);
	addParameter(p, 'type', 'amplitude', @ischar);

	parse(p, varargin{:});

	% Compute 'x' based on dB or raw contrast.
	if(p.Results.db)
		x = log10(mu_t / sig_b);

		% Compute magnitude or raw value of contrast based on user input.
		switch p.Results.type
			case 'amplitude'
				snr = real(20 * x);
			case 'power'
				snr = abs(10 * x);
		end
	else
		snr = mu_t / sig_b;
	end
end
