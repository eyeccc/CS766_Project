  function []=createFeatureMatrix()
% this function is a sample for using the feature extraction code to 
% extract artistic painting image features in batch
% and save the extracted image feature data to "feature_sample.mat"
% hit the run button to start now !

% remember to add the simplied graph-based visual saliency package to the path
 addpath('simpsal\simpsal');

folder1 = 'scraped-images\vangogh\';
[nl_gogh,vangogh] = featureExt_batch(folder1);
folder2 = 'scraped-images\gauguin\';
[nl_gauguin, gauguin] = featureExt_batch(folder2);
folder3 = 'scraped-images\braque\';
[nl_braque, braque] = featureExt_batch(folder3);
folder4 = 'scraped-images\gris\';
[nl_gris, gris] = featureExt_batch(folder4);
folder5 = 'scraped-images\monet\';
[nl_monet, monet] = featureExt_batch(folder5);
folder6 = 'scraped-images\raphael\';
[nl_raphael, raphael] = featureExt_batch(folder6);
folder7 = 'scraped-images\titian\';
[nl_titian, titian] = featureExt_batch(folder7);

clear folder1 folder2 folder3 folder4 folder5 folder6 folder7  
save('feature_sample.mat');

