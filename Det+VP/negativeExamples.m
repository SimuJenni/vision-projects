function [ neg ] = negativeExamples( VOCdevkit )
% negative examples from VOC train-set 

pascal_init;
cls = 'car';

ids = textread(sprintf(VOCopts.imgsetpath, 'train'), '%s');
neg = [];
numneg = 0;
   
% Parse negative examples
for i = 1:length(ids);
    if mod(i, 100) == 0
        fprintf('Parsing negatives: %d/%d\n', i, length(ids));
    end 
    rec = PASreadrecord(sprintf(VOCopts.annopath, ids{i}));
    clsinds = strmatch(cls, {rec.objects(:).class}, 'exact');
    if isempty(clsinds)
      numneg = numneg+1;
      neg(numneg).im = [VOCopts.datadir rec.imgname];
    end
end


end

