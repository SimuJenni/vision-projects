epflDatasetPath = '~/data/epfl-gims08/tripod-seq/';
files = dir([epflDatasetPath 'tripod_seq_' car_id '_*.jpg']);

factor = 3;
w = 64*factor;
h = 64*factor;
N = 2;%*factor;
W = w*(N+1);
H = h;
n = length(files);



rep  = zeros(H,W+w);
repbroom  = zeros(H,W);
times_fid = fopen( [epflDatasetPath 'times.txt'], 'r' );
times = cell(1,20);
i = 1;
while(1)
    times_ = fgets( times_fid );
    if( length(times_) < 2 )
        break;
    end
    times{1,i} = str2num(times_);
     i = i + 1;
end
    

front_indices = [50  75   2 107 23 46  64  93 28   6 61  13  68 109  4  10  63  88 51 54];
num_360 = [109 97 105 110 79 89 122 163 93 136 81 109 106 134 84 126 118 108 96 72];
rot_dir = [1 1 1 1 -1 1 1 1 1 -1 -1 -1 1 -1 -1 -1 1 -1 -1 1];

for car_id = 1 : 1
    
    files = dir([epflDatasetPath 'tripod_seq_' num2str(int32(car_id), '%02i') '_*.jpg']);
    times_ = times{car_id};
    deg_per_sec = 360/times_(num_360(car_id));
    
    bbox_fid = fopen([epflDatasetPath 'bbox_' num2str(int32(car_id), '%02i') '.txt']); 
    
    stride_per_deg = (W/2/180.0);
    
    for i = 1 : num_360(car_id)%length(files)
        
        bbox = int32(str2num(fgets(bbox_fid)));
        I = imread( [epflDatasetPath files(i).name] );
        I = I(bbox(2):bbox(2)+bbox(4), bbox(1):bbox(1)+bbox(3), : );
        theta = rot_dir(car_id)*(times_(i) - times_(front_indices(car_id)))*deg_per_sec;
        
        if( theta > 180 ) 
            theta = theta - 360;
        else if (theta < -180 )
                theta = theta + 360;
            end
        end
        
        
        position =  [1 50];
        J = insertText(I,position,theta,'AnchorPoint','LeftBottom');       
        figure(1), imshow(J);
%         pause(.01);

        pos = int32(max( [W/2,H/2] - [theta*stride_per_deg, 0],[1 1]));
        I = (im2double(rgb2gray(imresize(I,[w,h]))));
     
        rep( : , pos(1) : pos(1) + 20 ) =  rep( : , pos(1) : pos(1) + 20 ) + I(:,w/2-10:w/2+10);%rep( : , pos(1) : pos(1) + 10 - 1  ) + I; 
        
%         stride = floor((N/n)*w*(i-1)) + 1;
%         rep( 1:h, stride:stride+w-1) = rep(1:h, stride:stride+w-1) + imresize( I, [h w] );
%         im = imresize( I, [h w] );
%         repbroom( 1:h, floor(w/2)+stride:floor(w/2)+stride+10-1) = im(:,floor(w/2)+[1:10]);
        
        figure(2), imshow(rep,[]);
        pause(.01);
        
    end
    fclose(bbox_fid);
end





fclose(times_fid);