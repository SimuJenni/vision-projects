function [ VPresult, BBresult, vpPerformance, bbPerformance ] = ... 
    testPerformance( W, bbM, vpModel, featType, imScale, ...
    epflDatasetPath, startID, endID, visualize )
%VPresult and BBresult are the ratio of correctly classified
%viewpoints/bounding-boxes only considering the highest scoring match

[featExtractor, cellSize, featDim, visualizer] = getFeatExtractor( ...
    featType, imScale);
xpad = 2*size(W,2);
[test, frames] = epflData( epflDatasetPath, startID, endID );
vpPerformance = zeros(1, length(test));
bbPerformance = zeros(1, length(test));

num = 0;

for id = startID: endID
    fprintf('Testing: %d/%d\n', id, endID);
    for j = 1 : frames(id)
        idx = num+j;
        testFeat = featExtractor(imread(test{idx}.im));
        testPad = padFeature(testFeat, xpad, 0);
        score = convn(testPad, W, 'valid');
        [~, I] = max(score(:));
        [row, col] = ind2sub(size(score),I);
        maxRegion = testPad(row:row+size(W,1)-1, col:col+size(W,2)-1,:);
        scMap = sum(W.*maxRegion, 3);
        bbScore = cell(1, size(W,2));
        for i = 1:size(W,2)
            if(bbM(3,i)-bbM(1,i)<0)
                bbScore{i} = sum(sum(scMap(bbM(2,i):bbM(4,i),1:bbM(3,i))));
                bbScore{i} = bbScore{i}+sum(sum(scMap(bbM(2,i):bbM(4,i),...
                    bbM(1,i):end)));
            else
                bbScore{i} = sum(sum(scMap(bbM(2,i):bbM(4,i),...
                    bbM(1,i):bbM(3,i))));
            end
        end
        [~, I] = max(cell2mat(bbScore));
        predBB = mod(bbM(:,I)'+[col-xpad/2,row,col-xpad/2,row]-1, size(W,2));
        predBB = round(predBB*cellSize/imScale);
        if visualize
            box{1} = predBB;
            box{2} = test{idx}.bbox;
            showboxes(imread(test{idx}.im), box)
        end
        bbPerformance(idx) = box_overlap(predBB, test{idx}.bbox) > 0.5;
        vpPerformance(idx) = test{idx}.angle>=vpModel(1,I) && ...
            test{idx}.angle<=vpModel(2,I);
    end
    num = num+frames(id);
end
VPresult = sum(vpPerformance)/length(vpPerformance);
BBresult = sum(bbPerformance)/length(bbPerformance);

end



