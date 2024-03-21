%% Calculate contrast

% INPUTS:
%   mu_t            -> (numeric) mean of a target region
%   mu_b            -> (numeric) mean of a backgroud region

% INPUTS (optional):
%   amplitude       -> (logical) compute the amplitude contrast if true or the
%                      power contrast if false
%   magnitude       -> (logical) compute the magnitude of the contrast if true
%                      or the real value of the contrast if false
%                      power contrast if false
%   db              -> (logical) compute the contrast in dB if true or the
%                      raw contrast if false
% OUTPUT:
%   contrast        -> (dB) contrast measurement
function contrast = computeContrast(mu_t, mu_b, varargin)
	p = inputParser();

	if(~isnumeric(mu_t) || ~isnumeric(mu_b))
		error('Inputs "mu_t" and "mu_b" must be numeric.');
	end

	addParameter(p, 'amplitude', true, @islogical);
	addParameter(p, 'db', true, @islogical);
	addParameter(p, 'magnitude', true, @islogical);

	parse(p, varargin{:});

	% Select 'a' based on amplitude or power contrast.
	if(p.Results.amplitude)
		a = 20;
	else
		a = 10;
	end

	% Compute 'x' based on dB or raw contrast.
	if(p.Results.db)
		x = log10(mu_t / mu_b);
	else
		x = mu_t / mu_b;
	end

	% Compute magnitude or raw value of contrast based on user input.
	if(p.Results.magnitude)
		contrast = real(a * x);
	else
		contrast = abs(a * x);
	end
end
