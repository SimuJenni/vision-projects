function [ W, b ] = svmTrain( posTrain, negTrain, lambda, biasMult, oldW, oldB )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    train = [posTrain, negTrain];
    X = cell2mat(cellfun(@(x) double(x(:)), train, 'UniformOutput', false));
    y = [];
    y(1:length(posTrain)) = 1;
    y(end:end+length(negTrain)) = -1;
    fprintf('Training SVM... \n');
    weight = [];
    weight(1:length(posTrain)) = 1;
    weight(end:end+length(negTrain)) = length(posTrain)/length(negTrain);
    if nargin<5
        [w b] = vl_svmtrain(X, y, lambda, 'verbose', 'weights', weight,...
            'BiasMultiplier', biasMult, 'MaxNumIterations', 500000);
    else
        [w b] = vl_svmtrain(X, y, lambda, 'verbose','solver','sgd', ...
            'weights', weight, 'model', oldW(:), 'bias', oldB,...
            'BiasMultiplier', biasMult, 'MaxNumIterations', 500000);
    end
    W = reshape(w, size(posTrain{1}));

end

