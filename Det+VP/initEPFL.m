function [pos, bbModel, vpModel] = initEPFL( epflPath, startID,...
    endID, stretchFactor, sigma, imScale, featType, visualize )
%Extracts positive training examples from the EPFL-dataset and initializes
%the bb and vp model. Results in one example per image. Should be used when
%additional training examples from other datasets are used.

% Get relevant data and feature info
[train, frames, ~, w, h] = epflData( epflPath, startID, endID );
[featExtractor, cellSize, featDim, visualizer] = getFeatExtractor( ...
    featType, imScale);

lowerBounds = round(cellfun(@(x) x.bbox(4), train)/cellSize*imScale);
bbHeights = round(cellfun(@(x) x.bbox(4)-x.bbox(2), train)/cellSize*imScale);
resHeight = max(bbHeights);

% Set dimensions
f = featExtractor(imread(train{1}.im));
wFeat = size(f,2);
hFeat = size(f,1);
fSize = round(wFeat/2);     % Importance-filter size
W = round(stretchFactor*wFeat+fSize);
H = hFeat;

% Importance filter
g_x = fspecial('gaussian',[1 fSize], sigma/cellSize);
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
    bbHeight = bbHeights(num+1:num+frames(id));

    for i = 1 : frames(id)
        
        unfold  = zeros(H, W, featDim);  % to test
        
        % Compute the stride based on the angle
        idx = num+i;
        stride = floor(stretchFactor*wFeat*(1/2+train{idx}.angle/360)) + 1;
        
        % Associate bounding-box info with X-position of model
        yOs = lowerBound(i)-resHeight;
        xOs = stride-fSize;
        featScale = imScale/cellSize;
        bbXmin{stride}(end+1) = round(train{idx}.bbox(1)*featScale)+xOs;
        bbXmax{stride}(end+1) = round(train{idx}.bbox(3)*featScale)+xOs;
        bbYmin{stride}(end+1) = round(train{idx}.bbox(2)*featScale)-yOs;
        bbYmax{stride}(end+1) = round(train{idx}.bbox(4)*featScale)-yOs;
        vp{stride}(end+1) = ceil(train{idx}.angle);
        vp{stride}(end+1) = floor(train{idx}.angle);

        % Compute image-features
        feat = featExtractor(imread(train{idx}.im));

        % Extract relevant segment, weight with filter and stitch
        seg = feat(:,fSize+round(-fSize/2:fSize/2-1),:).*g_x;
        unfold(lowerBound(i)+(-bbHeight(i):0), stride+(0:fSize-1),:) = ...
            unfold(lowerBound(i)+(-bbHeight(i):0), stride+(0:fSize-1),:)+...
            seg(lowerBound(i)+(-bbHeight(i):0),:,:);  
        
        % Extract relevant region and trim the result
        reg = unfold(max(lowerBound)-resHeight+(2:resHeight), :, :);
        pos{idx} = reg(:, fSize/2+1:end-fSize/2, :);
        pos{idx} = pos{idx}*size(pos{idx},2);
        if visualize
            figure(1);
            visualizer(pos{idx});
        end
    end
    num = num+frames(id);

end

% Take the average bb-positions and assotiate with X-position of the model
resW = size(pos{1},2);
bbModel = zeros(4,resW);
bbModel(1,:) = mod(round(cellfun(@mean, bbXmin(1:resW))), resW);
bbModel(2,:) = max(round(cellfun(@mean, bbYmin(1:resW)))-1,1);
bbModel(3,:) = mod(round(cellfun(@mean, bbXmax(1:resW))), resW);
bbModel(4,:) = max(round(cellfun(@mean, bbYmax(1:resW)))-1,1);
bbModel(bbModel==0) = 1;
vpModel = zeros(2,resW);
if(~strcmp('imgrad', featType))
    vpModel(1,:) = cellfun(@min, vp(1:resW));
    vpModel(2,:) = cellfun(@max, vp(1:resW));
end
for i=1:resW-1
    if vpModel(2,i)<vpModel(2,i+1)
        vpModel(2,i) = vpModel(2,i+1);
    end
end
vpModel(1,1) = -180;
vpModel(2,end) = 180;

end

