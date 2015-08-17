function [ result ] = padFeature( feat, wPad, hPad)
%Adds padding to the featuremap

featDim = size(feat, 3);
result = zeros(size(feat,1)+hPad, size(feat,2)+wPad, featDim);
result(round(hPad/2)+1:round(hPad/2)+size(feat,1),...
    round(wPad/2)+1:round(wPad/2)+size(feat,2), :) = feat; 

end

