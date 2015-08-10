
car_id = '01';
epflDatasetPath = '~/data/epfl-gims08/tripod-seq/';

files = dir([epflDatasetPath 'tripod_seq_' car_id '_*.jpg']);
factor = 6;
w = 64*factor;
h = 64*factor;
N = 2;%*factor;
W = w*(N+1);
H = h;
n = length(files);

rep  = zeros(H,W);
repbroom  = zeros(H,W);

for i = 1 : length(files)
    
    stride = floor((N/n)*w*(i-1)) + 1;
    I = im2double(rgb2gray(imread( [epflDatasetPath files(i).name] )));
    I = imgradient(I);
    rep( 1:h, stride:stride+w-1) = rep(1:h, stride:stride+w-1) + imresize( I, [h w] );
    im = imresize( I, [h w] );
    repbroom( 1:h, floor(w/2)+stride:floor(w/2)+stride+10-1) = im(:,floor(w/2)+[1:10]);
    
    figure(1), imshow(repbroom,[]);
    pause(.001);
end

%figure(2), imshow(repbroom,[]);
