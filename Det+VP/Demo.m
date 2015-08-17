run ~/3rd_party_libs/vlfeat-0.9.19/toolbox/vl_setup
epflDatasetPath = '~/data/epfl-gims08/tripod-seq/';

stretchFactor = 1.5;                % Stretching coefficient 
sigma = 16/cellSize;     % Parameter for importance filter
imSizeFactor = 0.5; 
visualize = true;

% Get relevant data and feature info
startID = 1;
endID = 2;
pos = posExamplesFrom(epflDatasetPath, startID, endID, stretchFactor, sigma,...
    imSizeFactor, visualize );


