function [posTrain, bbModel, vpModel] = posExamplesEPFL( epflPath, startID,...
    endID, stretchFactor, sigma, imSizeFactor, featType, visualize )
%Extracts positive training examples from the EPFL-dataset and initializes
%the bb-model

% Get relevant data and feature info
[train, frames, times, w, h] = epflData( epflPath, startID, endID );
[featExtractor, cellSize, featDim, visualizer] = getFeatExtractor(featType, imSizeFactor);

lowerBounds = round(cellfun(@(x) x.bbox(4), train)/cellSize*imSizeFactor);
bbHeights = cellfun(@(x) x.bbox(4)-x.bbox(2), train);
resHeight = floor(max(bbHeights)/cellSize*imSizeFactor);

% Set dimensions
wFeat = round(w/cellSize*imSizeFactor);
hFeat = round(h/cellSize*imSizeFactor);
filterSize = round(wFeat/2);
W = round(stretchFactor*wFeat+filterSize);
H = hFeat;

% Importance filter
g_x = fspecial('gaussian',[1 filterSize], sigma);
g_x = repmat(g_x, [hFeat, 1, featDim]);

posTrain = cell(1, endID-startID);
num = 0;

% To cache bb-positions and viewpoints
bbXmin = cell(1, round(stretchFactor*wFeat)+1);
bbXmax = cell(1, round(stretchFactor*wFeat)+1);
bbYmin = cell(1, round(stretchFactor*wFeat)+1);
bbYmax = cell(1, round(stretchFactor*wFeat)+1);
vp = cell(1, round(stretchFactor*wFeat)+1);

for id = startID: endID
    fprintf('Positive examples: %d/%d\n', id, endID);
    unfold  = zeros(H, W, featDim);
    lowerBound = max(lowerBounds(num+1:num+frames(id)), resHeight);

    for i = 1 : frames(id)
        % Compute the stride based on the angle
        idx = num+i;
        stride = floor(filterSize+stretchFactor*wFeat*train{idx}.angle/360) + 1;
        
        % Associate bounding-box info with X-position of model
        bbXmin{stride+filterSize/2}(end+1) = round(train{idx}.bbox(1)/cellSize);
        bbXmax{stride+filterSize/2}(end+1) = round(train{idx}.bbox(3)/cellSize);
        bbYmin{stride+filterSize/2}(end+1) = round(train{idx}.bbox(2)/cellSize);
        bbYmax{stride+filterSize/2}(end+1) = round(train{idx}.bbox(4)/cellSize);
        vp{stride+filterSize/2}(end+1) = round(train{idx}.angle);


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
    posTrain{id} = unfold(:, filterSize/2+1:end-filterSize/2, :);
    posTrain{id}(:, 1:filterSize/2, :) = posTrain{id}(:, 1:filterSize/2, :)...
        + unfold(:, end-filterSize/2+1:end, :);
    posTrain{id}(:, end-filterSize/2+1:end, :) = posTrain{id}(:, end-filterSize/2+1:end, :)...
        + unfold(:, 1:filterSize/2, :);
    if visualize
        figure(2), visualizer(posTrain{id});
    end
end

% Take the average bb-positions and assotiate with X-position of the model
resW = size(posTrain{1},2);
bbModel = zeros(4,resW);
bbModel(1,:) = round(cellfun(@mean, bbXmin(1:resW)));
bbModel(2,:) = round(cellfun(@mean, bbYmin(1:resW)));
bbModel(3,:) = round(cellfun(@mean, bbXmax(1:resW)));
bbModel(4,:) = round(cellfun(@mean, bbYmax(1:resW)));
vpModel = zeros(2,resW);
vpModel(1,:) = cellfun(@min, vp(1:resW));
vpModel(2,:) = cellfun(@max, vp(1:resW));

end

