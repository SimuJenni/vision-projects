function result = posExamplesFrom( epflPath, startID, endID, stretchFactor,...
    sigma, imSizeFactor )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

% Get relevant data and feature info
[train, frames, times, w, h] = epflData( epflPath, startID, endID );
[featExtractor, cellSize, featDim, visualizer] = getFeatExtractor('hog');

lowerBound = round(max(cellfun(@(x) x.bbox(4), train))/cellSize*imSizeFactor);
bbHeights = cellfun(@(x) x.bbox(4)-x.bbox(2), train);
resHeight = round(max(bbHeights)/cellSize*imSizeFactor);

% Init descriptor
wFeat = round(w/cellSize*imSizeFactor);
hFeat = round(h/cellSize*imSizeFactor);
filterSize = round(wFeat/2);
W = round(stretchFactor*wFeat+filterSize);
H = hFeat;

% Importance filter
g_x = fspecial('gaussian',[1 filterSize], sigma);
g_x = repmat(g_x, [hFeat, 1, featDim]);

result = cell(endID-startID);

for id = startID: endID
    unfold  = zeros(H, W, featDim);
    for i = 1 : frames(id)
        % Compute the stride based on the angle
        stride = floor(filterSize+stretchFactor*wFeat*train{i}.angle/360) + 1;

        % Compute image-features
        im = imresize( imread(train{i}.im), [round(h*imSizeFactor) nan] );
        feat = featExtractor(im);

        % Extract relevant segment, weight it according to filter and stitch
        seg = feat(:,filterSize+[-filterSize/2:filterSize/2-1],:).*g_x;
        unfold(:, filterSize+stride+[-filterSize/2:filterSize/2-1],:) = ...
            unfold(:, filterSize+stride+[-filterSize/2:filterSize/2-1],:)+seg;  

        % Visualize the construction
        figure(1), visualizer(unfold);
        pause(.0001);
    end
    
    % Extract relevant region and trim the result
    unfold = unfold(lowerBound-resHeight:lowerBound, :, :);
    result{id} = unfold(:, filterSize/2:end-filterSize/2, :);
    result{id}(:, 1:filterSize/2, :) = result{id}(:, 1:filterSize/2, :)...
        + unfold(:, end-filterSize/2+1:end, :);
    result{id}(:, end-filterSize/2+1:end, :) = result{id}(:, end-filterSize/2+1:end, :)...
        + unfold(:, 1:filterSize/2, :);
    figure(2), visualizer(result{id});
end

end

