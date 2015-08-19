run ~/3rd_party_libs/vlfeat-0.9.19/toolbox/vl_setup
run ~/3rd_party_libs/matconvnet/matlab/vl_setupnn
epflDatasetPath = '~/data/epfl-gims08/tripod-seq/';
VOCDevkitPath = '~/data/VOCdevkit';

featType = 'hog';
imSizeFactor = 0.6; 
[featExtractor, cellSize, featDim, visualizer] = getFeatExtractor(...
    featType, imSizeFactor);
stretchFactor = 1.2;     
sigma = 16/cellSize;     % Parameter for (gaussian) importance filter 
visualize = true;
lambda = 0.1;
maxNeg = 10000;
hardNegItr = 4;

% Get relevant data and feature info
startID = 1;
endID = 16;
[posTrain, bbModel, vpModel] = initModelEPFL(epflDatasetPath, startID,...
    endID, stretchFactor, sigma, imSizeFactor, featType, visualize );
neg = negativeExamples(VOCDevkitPath);

% Get negative training-examples
negTrain = randomNegatives(neg(1:500), size(posTrain{1}), featType, maxNeg);

% Learn SVM classifier
[W, b] = svmTrain(posTrain, negTrain, lambda);
for i=1:hardNegItr
    % Get hard negative examples
    newNeg = hardNegatives(neg, W, b, featType, maxNeg, -1.05);
    if(length(newNeg)<0.1*maxNeg)
        break;
    end
    negTrain(1:length(newNeg)) = newNeg;
    [W, b] = svmTrain(posTrain, negTrain, lambda, W, b);
end

% Test demo
startID = 17;
endID = 20;
VPresult = testVPPerformance(W, bbModel, vpModel, featExtractor,...
    epflDatasetPath, startID, endID);