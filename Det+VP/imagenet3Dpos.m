function pos = imagenet3Dpos(Pascal3D, flippedpos)
% Parses positive training images from imagenet including annotations.

cls = 'car';
fileID = fopen([Pascal3D '/Image_sets/' cls '_imagenet_train.txt']);
C = textscan(fileID, '%s');
fclose(fileID);
ids=C{1};
N = length(ids);
pos = cell(1, N);
path_image = sprintf([Pascal3D '/Images/%s_imagenet'], cls);
path_anno = sprintf([Pascal3D '/Annotations/%s_imagenet'], cls);
count = 0;

for i = 1:N        
    file_ann = sprintf('%s/%s.mat', path_anno, char(ids(i)));
    if mod(i, 500) == 0
        fprintf('%s: parsing imagenet3D positives: %d/%d\n', cls, i, length(ids));
    end    
    try
        image = load(file_ann);
    catch
        continue;
    end
    record = image.record;
    objects = record.objects;
    numInstances=0;
    for j = 1:length(objects)
        numInstances=numInstances+strcmp(cls, objects(j).class);
    end
    for j = 1:length(objects)
        if isfield(objects(j), 'viewpoint') == 0 ||...
                ~strcmp(cls, objects(j).class) || ...
                objects(j).difficult == 1 || ...
                objects(j).truncated || ...
                objects(j).occluded
            continue;
        end
        viewpoint = objects(j).viewpoint;
        if isempty(viewpoint) || abs(viewpoint.elevation)>20
            continue;
        end
        bbox = objects(j).bbox;
        file_img = sprintf('%s/%s.JPEG', path_image, char(ids(i)));
        count = count + 1;
        pos{count}.im = file_img;
        pos{count}.bbox = bbox;
        
        if viewpoint.distance == 0
            azimuth = viewpoint.azimuth_coarse-360;
        else
            azimuth = viewpoint.azimuth-360;
        end
        if azimuth <= -180
            azimuth = 360 + azimuth;
        end
        
        pos{count}.angle = azimuth;

        % flip the positive example
        if flippedpos
            oldx1 = bbox(1);
            oldx2 = bbox(3);
            bbox(1) = record.imgsize(1) - oldx2 + 1;
            bbox(3) = record.imgsize(1) - oldx1 + 1;
            count = count + 1;
            pos{count}.im = file_img;
            pos{count}.bbox = bbox;
            % flip viewpoint
            azimuth = 360 - azimuth;
            if azimuth >= 360
                azimuth = 360 - azimuth;
            end
            pos(count).angle = azimuth;     
        end
    end
end

pos = pos(1:count);
