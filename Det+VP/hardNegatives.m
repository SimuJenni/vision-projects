function [ negTrain ] = hardNegatives(neg, W, b, featType, maxnum, thresh)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

featExtractor = getFeatExtractor(featType, 1);
negTrain = [];

for i=1:length(neg)
    if(mod(i,20)==1)
        fprintf('Hard negatives: %d/%d (%d/%d) \n', length(negTrain),...
            maxnum, i, length(neg));
    end
    feat = featExtractor(imresize(imread(neg(i).im),2));
    feat = padFeature(feat, size(W, 2), 0);

    score = convn(feat, W, 'valid') + b;
    [row, col] = find(score>thresh);
    for j=1:length(row)
        negTrain{end+1} = feat(row(j):row(j)+size(W,1)-1, col(j):col(j)+size(W,2)-1, :);
    end
    if(length(negTrain)>maxnum)
        break;
    end
end

end

