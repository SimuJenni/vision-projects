function [ train ] = epflData( epflDatasetPath, startIdx, endIdx )
%PARSES TRAINING DATA FROM THE EPFL DATASET
%   Returns a cell-array of training examples taken from the sequences
%   startIdx until endIdx. 
%   Training examples have the following fields:
%       .im: Image-path
%       .bbox: Bounding box in the format [xmin, ymin, xmax, ymax]
%       .angle: Viewpoint angle in radians

% Gather info about the sequences
info_fid = fopen( [epflDatasetPath 'tripod-seq.txt'], 'r' );
infL = str2num(fgets(info_fid));
numSeq = infL(1); imWidth = infL(2); imHeight = infL(3);
numFrames = str2num(fgets(info_fid));
imgFormat = fgets(info_fid);
bbFormat = strtrim(fgets(info_fid));
frames360 = str2num(fgets(info_fid));
frontFrame = str2num(fgets(info_fid));
rotDir = str2num(fgets(info_fid));

% Gather frame-times
times_fid = fopen( [epflDatasetPath 'times.txt'], 'r' );
times = []
idx = 1;
while ~feof(times_fid)
   times{idx} = str2num(fgets(times_fid));
   idx = idx+1;
end
 
% Parse the training data
train = [];
idx = 1;
for seq = startIdx:endIdx
    bbox_fid = fopen(sprintf([epflDatasetPath bbFormat], seq), 'r');
    for frame = 1:frames360(seq)
        train{idx}.im = sprintf([epflDatasetPath imgFormat], seq, frame);
        train{idx}.seq = seq;
        train{idx}.frame = frame;
        train{idx}.angle = computeAngle(times{seq}, frame, ...
            frontFrame(seq), rotDir(seq), frames360(seq));
        train{idx}.bbox = str2num(fgets(bbox_fid));
        idx = idx+1;
    end
end

end

