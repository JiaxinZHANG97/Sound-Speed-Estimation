function [gcnr] = computeGcnr(...
		s_i, s_o, varargin)
	% computeGcnr Compute the measured or predicted gCNR, depending the inputs.
	%
	% Inputs:
	% -------
	% s_i
	%     -> (numeric) matrix of values from your target region
	% s_o
	%     -> (numeric) matrix of values from your background region
	% binSelectionMethod
	%     -> (string) options are
	%        1. 'manual' (default)
	%        2. 'scott1979optimal'
	%        3. 'wand1997data'
	% numBins
	%     -> (numeric) number of bins in the histogram (default = 32)
	%
	% Outputs:
	% --------
	% gcnr     -> calculated gCNR
	% s_i_hist -> histogram for target region
	% s_o_hist -> histogram for background reion
	% x_hist   -> vector containing the centers of histogram bins
	%
	% Revisions:
	% ----------
	% 1.0 -> Mardava Gubbi
	p = inputParser();

	addParameter(p, 'predicted', false, @islogical);

	% Parameters required only for gCNR prediction framework.
	addParameter(p, 'targetDistribution', 'gamma', @ischar);
	addParameter(p, 'backgroundDistribution', 'exponential', @ischar);

	p.KeepUnmatched = true;
	parse(p, varargin{:});

	if p.Results.predicted
		target_params = estimateDistributionParams(...
			s_i, 'distribution', p.Results.targetDistribution);
		background_params = estimateDistributionParams(...
			s_o, 'distribution', p.Results.backgroundDistribution);
		[gcnr, ~, ~, ~, ~] = getGcnrTheoretical(...
			'targetParams', target_params, ...
			'backgroundParams', background_params, varargin{:});
	else
		[gcnr, ~, ~, ~] = getGcnrExperimental(s_i, s_o, varargin{:});
	end
end
