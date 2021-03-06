function [ negTrain ] = randomNegatives( neg, featSize, featType, maxnum )
%Extracts random negative training examples

featureExtractor = getFeatExtractor(featType, 2);
numneg = length(neg);
rndneg = ceil(maxnum/numneg);
negTrain = cell(0);
for i = 1:min(numneg, maxnum)
  if mod(i,200) == 0  
    fprintf('Random negatives: %d/%d\n', i, min(numneg, maxnum));
  end
  feat=featureExtractor(imread(neg(i).im));

  if size(feat,2) > featSize(2) && size(feat,1) > featSize(1)
    for j = 1:rndneg
      x = randi(size(feat,2)-featSize(2)+1);
      y = randi(size(feat,1)-featSize(1)+1);
      negTrain{end+1} = feat(y:y+featSize(1)-1, x:x+featSize(2)-1,:);
    end
  end
end


end

