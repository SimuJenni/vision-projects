run ~/3rd_party_libs/vlfeat-0.9.19/toolbox/vl_setup
epflDatasetPath = '~/data/epfl-gims08/tripod-seq/';
VOCDevkitPath = '~/data/VOCdevkit';

featType = 'hog';
imSizeFactor = 0.5; 
[featExtractor, cellSize, featDim, visualizer] = getFeatExtractor(featType, imSizeFactor);
stretchFactor = 1.5;                % Stretching coefficient 
sigma = 24/cellSize;     % Parameter for importance filter
visualize = false;

% Get relevant data and feature info
startID = 1;
endID = 16;
[posTrain, wFeat, hFeat] = posExamplesEPFL(epflDatasetPath, startID,...
    endID, stretchFactor, sigma, imSizeFactor, featType, visualize );
neg = negativeExamples(VOCDevkitPath);
negTrain = randomNegatives(neg, size(posTrain{1}), featExtractor, 40000 );
train = [posTrain, negTrain];

X = cell2mat(cellfun(@(x) single(x(:)), train, 'UniformOutput', false));
y = [];
y(1:length(posTrain)) = 1;
y(end:end+length(negTrain)) = -1;
lambda = 0.01;
[w b] = vl_svmtrain(X, y, lambda, 'verbose');
W = reshape(w, size(posTrain{1}));
figure(3), visualizer(W);

testIdx = 17;
[test, frames, times, w, h] = epflData( epflDatasetPath, testIdx, testIdx );
testFeat = featExtractor(imread(test{50}.im));
testPadded = padFeature(testFeat, 2*wFeat, 0);
score = convn(testPadded, W, 'valid')+b;
imagesc(score);
