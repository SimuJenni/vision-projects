function [pos, bbModel, vpModel] = initModelEPFL( epflPath, startID,...
    endID, stretchFactor, sigma, imScale, featType, visualize )
%Extracts positive training examples from the EPFL-dataset and initializes
%the bb-model

% Get relevant data and feature info
[train, frames, ~, w, h] = epflData( epflPath, startID, endID );
[featExtractor, cellSize, featDim, visualizer] = getFeatExtractor( ...
    featType, imScale);

lowerBounds = round(cellfun(@(x) x.bbox(4), train)/cellSize*imScale);
bbHeights = cellfun(@(x) x.bbox(4)-x.bbox(2), train);
resHeight = round(max(bbHeights)/cellSize*imScale);

% Set dimensions
wFeat = ceil(w/cellSize*imScale);
hFeat = ceil(h/cellSize*imScale);
fSize = round(wFeat/2);     % Importance-filter size
W = round(stretchFactor*wFeat+fSize);
H = hFeat;

% Importance filter
g_x = fspecial('gaussian',[1 fSize], sigma);
g_x = repmat(g_x, [hFeat, 1, featDim]);

pos = cell(1, endID-startID);
num = 0;

% To cache bb-positions and viewpoints
bbXmin = cell(1, W-fSize+1);
bbXmax = cell(1, W-fSize+1);
bbYmin = cell(1, W-fSize+1);
bbYmax = cell(1, W-fSize+1);
vp = cell(1, round(stretchFactor*wFeat)+1);

for id = startID: endID
    fprintf('Positive examples: %d/%d\n', id, endID);
    unfold  = zeros(H, W, featDim);
    lowerBound = max(lowerBounds(num+1:num+frames(id)), resHeight);

    for i = 1 : frames(id)
        % Compute the stride based on the angle
        idx = num+i;
        stride = floor(stretchFactor*wFeat*(1/2+train{idx}.angle/360)) + 1;
        
        % Associate bounding-box info with X-position of model
        yOs = lowerBound(i)-resHeight;
        xOs = stride-fSize/2;
        featScale = imScale/cellSize;
        bbXmin{stride}(end+1) = round(train{idx}.bbox(1)*featScale)+xOs;
        bbXmax{stride}(end+1) = round(train{idx}.bbox(3)*featScale)+xOs;
        bbYmin{stride}(end+1) = round(train{idx}.bbox(2)*featScale)-yOs;
        bbYmax{stride}(end+1) = round(train{idx}.bbox(4)*featScale)-yOs;
        vp{stride}(end+1) = round(train{idx}.angle);

        % Compute image-features
        feat = featExtractor(imread(train{idx}.im));

        % Extract relevant segment, weight with filter and stitch
        seg = feat(:,fSize+round(-fSize/2:fSize/2-1),:).*g_x;
        unfold(yOs+1:resHeight, stride+(0:fSize-1),:) = ...
            unfold(yOs+1:resHeight, stride+(0:fSize-1),:)+...
            seg(yOs+1:resHeight,:,:);  

        if visualize
            % Visualize the construction
            figure(1), visualizer(unfold);
            pause(.0001);
        end
    end
    num = num+frames(id);
    
    % Extract relevant region and trim the result
    unfold = unfold(lowerBound-resHeight+1:lowerBound, :, :);
    pos{id} = unfold(:, fSize/2+1:end-fSize/2, :);
    pos{id}(:, 1:fSize/2, :) = pos{id}(:, 1:fSize/2, :) + unfold(:,...
        end-fSize/2+1:end, :);
    pos{id}(:, end-fSize/2+1:end,:) = pos{id}(:,end-fSize/2+1:end,:) + ...
        unfold(:, 1:fSize/2, :);
    if visualize
        figure(2), visualizer(pos{id});
    end
end

% Take the average bb-positions and assotiate with X-position of the model
resW = size(pos{1},2);
bbModel = zeros(4,resW);
bbModel(1,:) = mod(round(cellfun(@mean, bbXmin(1:resW))), resW);
bbModel(2,:) = round(cellfun(@mean, bbYmin(1:resW)));
bbModel(3,:) = mod(round(cellfun(@mean, bbXmax(1:resW))), resW);
bbModel(4,:) = round(cellfun(@mean, bbYmax(1:resW)));
bbModel(bbModel==0) = 1;
vpModel = zeros(2,resW);
vpModel(1,:) = cellfun(@min, vp(1:resW));
vpModel(2,:) = cellfun(@max, vp(1:resW));

end

