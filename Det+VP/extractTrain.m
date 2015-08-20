function train = extractTrain( W, bbModel, vpModel, pos, featType, imScale, sigma)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

[featExtractor, cellSize, featDim, visualizer] = getFeatExtractor( ...
    featType, imScale);

train = cell(0);

for i = 1:length(pos)
    if(mod(i, 100)==1)
        fprintf('New positives: %d/%d\n (added: %d)', i, length(pos), length(train));
    end
    angle = pos{i}.angle;
    I = and(vpModel(1,:)<=angle, vpModel(2,:)>angle);
    bbF = bbModel(:,I);
    if bbF(3)-bbF(1)>=0
        wF = bbF(3)-bbF(1);
    else
        wF1 = bbF(3);
        wF2 = size(W,2)-bbF(1);
        wF = wF1+wF2;
    end
    hF = bbF(4)-bbF(2);
    bbox = pos{i}.bbox;
    w = round((bbox(3)-bbox(1))*imScale/cellSize);
    h = round((bbox(4)-bbox(2))*imScale/cellSize);
    scale = sqrt(wF*hF/(w*h));
    bbCenter = round([bbox(2)+bbox(4), bbox(1)+bbox(3)]/2*scale*imScale/cellSize);
    if bbCenter(1)-round(hF/2) == 0
        bbCenter(1) = bbCenter(1)+1;
    end
    if bbCenter(2)-round(wF/2) == 0
        bbCenter(2) = bbCenter(2)+1;
    end
    try 
        feat = featExtractor(imresize(imread(pos{i}.im), scale));
        example = zeros(size(W));
        % Importance filter
        g_x = fspecial('gaussian',[1, wF+1], sigma/cellSize);
        g_x = repmat(g_x, [hF+1, 1, featDim]);
%         example = feat(1:size(W,1),1:size(W,2),:);
        if bbF(3)-bbF(1)>=0
            example(bbF(2):bbF(4),bbF(1):bbF(3), :) = feat(bbCenter(1)+round(-hF/2:hF/2), bbCenter(2)+round(-wF/2:wF/2),:).*g_x; 
        else
            example(bbF(2):bbF(4),end+(-wF2+1:0), :) = ... 
                feat(bbCenter(1)+round(-hF/2:hF/2), round(bbCenter(2)-wF/2)+(1:wF2),:).*g_x(:,1:wF2,:); 
            example(bbF(2):bbF(4),1:wF1, :) = ... 
                feat(bbCenter(1)+round(-hF/2:hF/2), round(bbCenter(2)-wF/2)+wF2+(1:wF1),:).*g_x(:,end+(-wF1+1:0),:);
        end
        train{end+1} = example;
%         visualizer(example);
    catch 
    end
end



end

