function [ featExtractor, cellSize, featDim, visualizer ] = ... 
    getFeatExtractor( featType, resizeFact )
%Returns a feature-extractor and its corresponding cellsize.
% Input: featType - 'hog' for HOG and 'imgrad' for image-gradients 
if(strcmp('hog', featType))
    cellSize = 8;
    featDim = 31;
    featExtractor = @(x) vl_hog(single(imresize(x,resizeFact)), cellSize);
    visualizer = @(x) imshow(vl_hog('render', single(x)),[]);
end
if(strcmp('imgrad', featType))
    cellSize = 1;
    featDim = 1;
    featExtractor = @(x) imgradient(im2double(rgb2gray(imresize(x,resizeFact))));
    visualizer = @(x) imshow(x);
end
if(strcmp('cnn', featType))
    cellSize = 16;
    featDim = 256;
    featExtractor = cnnFeatureExtractor('~/data/', resizeFact);
    visualizer = @(x) imshow(x);
end


