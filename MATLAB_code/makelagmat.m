function lag = makelagmat(numrows,numcols,maxlag)
% MAKELAGMAT    Generates a cell array containing lag information
% A matrix that contains the lag number of each element with respect to
% every other element is output by this function. These values are
% stored in a NUMROWS*NUMCOLS by NUMROWS*NUMCOLS matrix. The output is a
% cell array containing the indices from the lag matrix that corresponds to
% the cell number, i.e. LAG{1} will contain all the indices of points in
% the lag matrix that are equal to 1, and LAG{2} for those that are equal
% to 2, etc.
%
% LAG = MAKELAGMAT(NUMROWS, NUMCOLS, MAXLAG) will return a cell array of
% dimensions (MAXLAG, 1). Each cell will contain the indices from the lag
% matrix that corresponds to the cell number.
% 
% LAG = MAKELAGMAT(NUMROWS, NUMCOLS) will assume that MAXLAG is set to
% the maximum possible value.

% Revision History
% 2011-05-09 dh65
%   Created function to be used with slsc_mex
if nargin == 2
    maxlag = round(sqrt(numrows^2+numcols^2))-1;
end

% Set up temp and lagmat
temp = false(numrows,numcols);
lagmat = zeros(numrows*numcols,numrows*numcols,'int32');

% Iterate through every element
for i = 1:numrows*numcols
    % "Turn on" the element so that bwdist can find all other elements'
    % distances from element i
    temp(i) = 1;
    lagdist = round(bwdist(temp));
    % Store information in lagmat
    lagmat(i,:) = lagdist(:);
    temp(i) = 0;
end
clear temp lagdist

% Transfer information from lagmat to lag
% Only save the upper triangle, since the matrix is symmetric
lag = cell(maxlag,1);
for i = 1:maxlag
    lag{i} = int32(find(triu(lagmat) == i));
end