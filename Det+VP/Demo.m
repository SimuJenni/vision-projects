run ~/3rd_party_libs/vlfeat-0.9.19/toolbox/vl_setup
epflDatasetPath = '~/data/epfl-gims08/tripod-seq/';

% Get relevant data and feature info
carID = 9;  
[train, frames, times, w, h] = epflData( epflDatasetPath, carID, carID );
[featExtractor, cellSize, featDim, visualizer] = getFeatExtractor('hog');

N = 1.5;                % Stretching coefficient 
sigma = 16/cellSize;     % Parameter for importance filter

% Init descriptor
wFeat = floor(w/cellSize);
hFeat = floor(h/cellSize);
filterSize = floor((wFeat+1)/2);
W = round(N*wFeat+filterSize);
H = hFeat;
repbroom  = zeros(H, W, featDim);

% Importance filter
g_x=fspecial('gaussian',[1 filterSize], sigma);
g_x = repmat(g_x, [hFeat, 1, featDim]);

for i = 1 : length(train)
    % Compute the stride based on the angle
    stride = floor(filterSize+N*wFeat*train{i}.angle/360) + 1;

    % Compute image-features
    im = imresize( imread(train{i}.im), [h nan] );
    feat = featExtractor(im);

    % Extract relevant segment, weight it according to filter and stitch
    seg = feat(:,filterSize+[-filterSize/2:filterSize/2-1],:).*g_x;
    repbroom(:, filterSize+stride+[-filterSize/2:filterSize/2-1],:) = ...
        repbroom(:, filterSize+stride+[-filterSize/2:filterSize/2-1],:)+seg;  
    
    % Visualize the construction
    figure(1), visualizer(repbroom);
    pause(.0001);
end

result = repbroom(:, filterSize/2:end-filterSize/2, :);
result(:, 1:filterSize/2, :) = result(:, 1:filterSize/2, :)...
    + repbroom(:, end-filterSize/2+1:end, :);
result(:, end-filterSize/2+1:end, :) = result(:, end-filterSize/2+1:end, :)...
    + repbroom(:, 1:filterSize/2, :);
figure(2), visualizer(result);

