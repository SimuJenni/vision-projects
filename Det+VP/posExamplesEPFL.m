function [result, wFeat, hFeat] = posExamplesEPFL( epflPath, startID,...
    endID, stretchFactor, sigma, imSizeFactor, featType, visualize )
%Extracts positive training examples from the EPFL-dataset

% Get relevant data and feature info
[train, frames, times, w, h] = epflData( epflPath, startID, endID );
[featExtractor, cellSize, featDim, visualizer] = getFeatExtractor(featType, imSizeFactor);

lowerBounds = ceil(cellfun(@(x) x.bbox(4), train)/cellSize*imSizeFactor);
bbHeights = cellfun(@(x) x.bbox(4)-x.bbox(2), train);
resHeight = floor(max(bbHeights)/cellSize*imSizeFactor);

% Init descriptor
wFeat = round(w/cellSize*imSizeFactor);
hFeat = round(h/cellSize*imSizeFactor);
filterSize = round(wFeat/2);
W = round(stretchFactor*wFeat+filterSize);
H = hFeat;

% Importance filter
g_x = fspecial('gaussian',[1 filterSize], sigma);
g_x = repmat(g_x, [hFeat, 1, featDim]);

result = cell(1, endID-startID);
num = 0;

for id = startID: endID
    fprintf('Positive examples: %d/%d\n', id, endID);
    unfold  = zeros(H, W, featDim);
    lowerBound = max(lowerBounds(num+1:num+frames(id)), resHeight);

    for i = 1 : frames(id)
        % Compute the stride based on the angle
        idx = num+i;
        stride = floor(filterSize+stretchFactor*wFeat*train{idx}.angle/360) + 1;

        % Compute image-features
        feat = featExtractor(imread(train{idx}.im));

        % Extract relevant segment, weight it according to filter and stitch
        seg = feat(:,filterSize+[-filterSize/2:filterSize/2-1],:).*g_x;
        unfold(:, filterSize+stride+[-filterSize/2:filterSize/2-1],:) = ...
            unfold(:, filterSize+stride+[-filterSize/2:filterSize/2-1],:)+seg;  

        if visualize
            % Visualize the construction
            figure(1), visualizer(unfold);
            pause(.0001);
        end
    end
    
    num = num+frames(id);
    % Extract relevant region and trim the result
    unfold = unfold(lowerBound-resHeight+1:lowerBound, :, :);
    result{id} = unfold(:, filterSize/2:end-filterSize/2, :);
    result{id}(:, 1:filterSize/2, :) = result{id}(:, 1:filterSize/2, :)...
        + unfold(:, end-filterSize/2+1:end, :);
    result{id}(:, end-filterSize/2+1:end, :) = result{id}(:, end-filterSize/2+1:end, :)...
        + unfold(:, 1:filterSize/2, :);
    if visualize
        figure(2), visualizer(result{id});
    end
end

end

