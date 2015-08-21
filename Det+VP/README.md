## Model ##

The model consists of the filter (unfolded representation of object), a bounding-box model (bbM) and a viewpoint-model (vpM).
The vpM relates each column of the filter with a viewing angle and the bbM relates each column with a bounding-box (bb) indicating the region of the filter that is occupied by the object when seen from that viewpoint.
The unfolding is done using segments of the object weighted by a vertical gaussian. The parameter sigma controls the overlap between neighboring views.
Features can be chosen to be one of HOG, CNN or image-gradient (only really for visualization).

## Inference ##

For inference, the featuremap of the image is convoluted with the filter and the highest scoring position is used as object hypothesis. The assumption for the inference is that the matching score of the filter-region corresponding to the correct bb will overweight the rest of the filter. This of course doesn't work with split bb which will have to be handled differently. 

The vp and bb are then inferred by searching for the highest scoring subregion of the filter (according to bbM).

## Learning ##

The model (bbM, vpM) is initialised on the EPFL dataset. The filter is first trained using random negative examples from VOC and then using hard negatives. I also added code to add additional positive examples from imagenet (only extracting relevant regions). 

## Testing ##
The test-code so far is very simple and only considers the highest scoring object hypothesis (no PR-curves yet).

## Requirements ##

### Datasets:
- [EPFL](http://cvlab.epfl.ch/data/pose)
- [Pascal VOC 2007](http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/)
- [Pascal3D+](http://cvgl.stanford.edu/projects/pascal3d.html)


### Software:
- [VL_Feat](http://www.vlfeat.org)
- [MatconvNet](http://www.vlfeat.org/matconvnet/)
