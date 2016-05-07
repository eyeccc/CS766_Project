1.Description: This code is used for extracting artistic image features descirbed in:

Ying Wang (Florence) and Masahiro Takatsuka. SOM based Artistic Style Visualization. IEEE International Conference on Multimedia & Expo (ICME'13). San Jose. USA, July, 2013.

Any question and bug report please email:

florence8627@vislab.net
---------------------------------------------------------------------------------------------------------------------------------
Please properly cite the above paper if you are to use this code for research purpose
----------------------------------------------------------------------------------------------------------------------------------

2. Pre-requisite: The following package is needed to run the code:
  
Graph-based visual saliency matlab package either GBVS or Simpsal

GBVS and Simpsal are both available at http://www.vision.caltech.edu/~harel/share/gbvs.php

For our code we used the simplified version Simpsal to extract composition features. Of course you can try to use GBVS in extracting composition features !

Simpsal is already included in the current folder. To use Simpsal, manually add the folder of Simpsal in the path or in the command line,  type:

addpath('simpsal\simpsal');
----------------------------------------------------------------------------------------------------------------------------------

3. to test the feature extraction of all the sample paintings located in \painting samples in batch, open: 

createFeatureMatrix.m 

and hit run. It will save features of all the images located within a folder into one variable and the name list (e.g. file names of images) into another variable.
----------------------------------------------------------------------------------------------------------------------------------
4. Description of important functions

featureExt_batch.m: batch feature extraction.

featureExtraction.m: extracting all the features of a single painting image.
----------------------------------------------------------------------------------------------------------------------------------

Glbcolor.m: extracting global color features in temperature, weight and contrast

Localcolor.m: extracting local color features in temperature, weight and contrast from hue segmented images

Composition.m: divding the saliency map into nine regions and extracting features from each section. Also it calls ShapeRep.m to calculate the eccentricity and rectratio of the most salient region (pixel saliency > 0.5)

Hardline.m: use Hough transform to decide four parameters ( mean length, mean slope, std slope and straight line ratio) of hard lines appeared in painting images.
