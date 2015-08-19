function [ W, b ] = svmTrain( posTrain, negTrain, lambda, oldW, oldB )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    fprintf('Training SVM... \n');
    biasMult = 5;
    train = [posTrain, negTrain];
    X = cell2mat(cellfun(@(x) double(x(:)), train, 'UniformOutput', false));
    y = [];
    y(1:length(posTrain)) = 1;
    y(end:end+length(negTrain)) = -1;
    weight = [];
    weight(1:length(posTrain)) = 1;
    weight(end:end+length(negTrain)) = length(posTrain)/length(negTrain);
    if nargin<4
        [w b] = vl_svmtrain(X, y, lambda, 'verbose', 'weights', weight,...
            'BiasMultiplier', biasMult);
    else
        [w b] = vl_svmtrain(X, y, lambda, 'verbose','solver','sgd', ...
            'weights', weight, 'model', oldW(:), 'bias', oldB,...
            'BiasMultiplier', biasMult);
    end
    W = reshape(w, size(posTrain{1}));

end

