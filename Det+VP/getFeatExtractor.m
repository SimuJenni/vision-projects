function [ featExtractor, cellSize, featDim, visualizer ] = getFeatExtractor( featType )
%Returns a feature-extractor and its corresponding cellsize.
% Input: featType - 'hog' for HOG and 'imgrad' for image-gradients 
if(strcmp('hog', featType))
    cellSize = 8;
    featDim = 31;
    featExtractor = @(x) vl_hog(single(x), cellSize);
    visualizer = @(x) imshow(vl_hog('render', single(x)),[]);
end
if(strcmp('imgrad', featType))
    cellSize = 1;
    featDim = 1;
    featExtractor = @(x) imgradient(im2double(rgb2gray(x)));
    visualizer = @(x) imshow(x);
end

