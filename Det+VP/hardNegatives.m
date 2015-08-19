function [ neg ] = hardNegatives(negPaths, W, b, featType, maxnum, thresh)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

featExtractor = getFeatExtractor(featType, 1);
neg = [];

for i=1:length(negPaths)
    if(mod(i,20)==1)
        fprintf('Hard negatives: %d/%d (%d/%d) \n', length(neg),...
            maxnum, i, length(negPaths));
    end
    feat = featExtractor(imresize(imread(negPaths(i).im),2));
    feat = padFeature(feat, 2*size(W, 2), 0);

    score = convn(feat, W, 'valid') + b;
    [row, col] = find(score>thresh);
    for j=1:length(row)
        neg{end+1} = feat(row(j)+(0:size(W,1)-1), col(j)+(0:size(W,2)-1),:);
    end
    if(length(neg)>maxnum)
        neg = neg(1:maxnum);
        break;
    end
end

end

