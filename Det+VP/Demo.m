run ~/3rd_party_libs/vlfeat-0.9.19/toolbox/vl_setup
car_id = '01';
epflDatasetPath = '~/data/epfl-gims08/tripod-seq/';

files = dir([epflDatasetPath 'tripod_seq_' car_id '_*.jpg']);
factor = 2;
w = 64*factor;
h = 64*factor;
N = 2;%*factor;
W = w*(N+1);
H = h;
n = length(files);
bin = 8;
cellSize = 8;
featDim = 31;


wFeat = w/cellSize;
hFeat = h/cellSize;
rep  = zeros(H, W, featDim);
repbroom  = zeros(H/cellSize, W/cellSize, featDim);
filterSize = wFeat/2;
g_x=fspecial('gaussian',[1 filterSize], 2);
g_x = repmat(g_x, [hFeat, 1, 31]);


for i = 1 : length(files)
    
%     stride = floor((N/n)*w*(i-1)) + 1;
    stride = floor((N/n)*wFeat*(i-1)) + 1;

    I = im2double(rgb2gray(imread( [epflDatasetPath files(i).name] )));
%     Igrad = imgradient(I);
%     rep( 1:h, stride:stride+w-1) = rep(1:h, stride:stride+w-1) + imresize( Igrad, [h w] );
%     im = imresize( I, [h w]);
%     repbroom( 1:h, floor(w/2)+stride:floor(w/2)+stride+bin-1) = ...
%         repbroom( 1:h, floor(w/2)+stride:floor(w/2)+stride+bin-1)+im(:,floor(w/2)+[1:bin]);
    
    im = single(imresize( I, [h w] ));
    hog = vl_hog(im, cellSize) ;

    seg = hog(:,floor(wFeat/2)+[-filterSize/2:filterSize/2-1],:).*g_x;
    repbroom( 1:hFeat, floor(wFeat/2)+stride+[-filterSize/2:filterSize/2-1],:) = ...
        repbroom( 1:hFeat, floor(wFeat/2)+stride+[-filterSize/2:filterSize/2-1],:)+seg;
    figure(1), imshow(vl_hog('render', single(repbroom)),[]);
    pause(.001);
end

% figure(2), imshow(repbroom,[]);

