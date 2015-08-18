run ~/3rd_party_libs/vlfeat-0.9.19/toolbox/vl_setup
epflDatasetPath = '~/data/epfl-gims08/tripod-seq/';
VOCDevkitPath = '~/data/VOCdevkit';

featType = 'hog';
imSizeFactor = 0.5; 
[featExtractor, cellSize, featDim, visualizer] = getFeatExtractor(featType, imSizeFactor);
stretchFactor = 1.5;                % Stretching coefficient 
sigma = 16/cellSize;     % Parameter for importance filter
visualize = false;
lambda = 0.001;
maxNeg = 50000;
hardNegItr = 3;

% Get relevant data and feature info
startID = 1;
endID = 16;
[posTrain, bbModel, vpModel] = posExamplesEPFL(epflDatasetPath, startID,...
    endID, stretchFactor, sigma, imSizeFactor, featType, visualize );
neg = negativeExamples(VOCDevkitPath);

% Get negative training-examples
negTrain = randomNegatives(neg, size(posTrain{1}), featType, maxNeg );

% Learn SVM classifier
[W, b] = svmTrain(posTrain, negTrain, lambda);
for i=1:hardNegItr
    % Get hard negative examples
    negTrain = hardNegatives(neg, W, b, featType, maxNeg, -1.0005);
    [W, b] = svmTrain(posTrain, negTrain, lambda, W, b);
end

% Test demo
testIdx = 20;
xpad = size(W,2);
[test, frames, times, w, h] = epflData( epflDatasetPath, testIdx, testIdx );
testFeat = featExtractor(imread(test{1}.im));
testPadded = padFeature(testFeat, xpad, 0);
score = convn(testPadded, W, 'valid');
imagesc(score);

[C, I] = max(score(:));
[I_row, I_col] = ind2sub(size(score),I);

