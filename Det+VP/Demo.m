run ~/3rd_party_libs/vlfeat-0.9.19/toolbox/vl_setup
run ~/3rd_party_libs/matconvnet/matlab/vl_setupnn
epflDatasetPath = '~/data/epfl-gims08/tripod-seq/';
Pascal3D = '~/data/PASCAL3D+_release1.1/';
VOCDevkitPath = '~/data/VOCdevkit';

featType = 'hog';    % one of 'imgrad', 'hog', 'cnn'
imScale = 0.8;    % scale image by this before feature computation
stretchFactor = 1.3;     
sigma = 24;     % Parameter for importance filter (divided by cellSize)
visualize = false;

% SVM parameters
lambda = 0.001;           
biasMult = 2;

maxNeg = 20000;
hardNegItr = 10;

% Get relevant data and feature info
startID = 1;
endID = 16;
[posTrain, bbModel, vpModel] = initEPFL(epflDatasetPath, startID,...
    endID, stretchFactor, sigma, imScale, featType, visualize );
% [posTrain, bbModel, vpModel] = unfoldInitEPFL(epflDatasetPath, startID,...
%     endID, stretchFactor, sigma, imScale, featType, visualize );
neg = negativeExamples(VOCDevkitPath);

% Get negative training-examples
negTrain = randomNegatives(neg(1:500), size(posTrain{1}), featType, maxNeg);

% Learn initial model
[W, b] = svmTrain(posTrain, negTrain, lambda, biasMult);

% Get additional positive training examples
pos = imagenet3Dpos(Pascal3D, false);
newPosTrain = extractTrain(W, bbModel, vpModel, pos, featType, 1, sigma);
posTrain = [posTrain, newPosTrain];
[W, b] = svmTrain(posTrain, negTrain, lambda, biasMult, W, b);

for i=1:hardNegItr
    % Get hard negative examples
    newNeg = hardNegatives(neg, W, b, featType, maxNeg, -1.002);
    if(length(newNeg)<maxNeg)
        break;
    end
    negTrain(1:length(newNeg)) = newNeg;
    [W, b] = svmTrain(posTrain, negTrain, lambda, biasMult, W, b);
end

% Test
startID = 17;
endID = 20;
[VPres, BBres] = testPerformance(W, bbModel, vpModel, featType, imScale,...
    epflDatasetPath, startID, endID, visualize);