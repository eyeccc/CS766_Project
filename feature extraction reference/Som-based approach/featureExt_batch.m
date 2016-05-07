function [namelist, feature] = featureExt_batch(folder)
% this function calls all the feature extraction functions
% perform feature extraction of all the images in a folder in batch
% input variables
%  folder---string, folder name
% output variables:
%  namelist--- cell array, the name list of all the paintings
%  feature---matrix, the extracted numeric image features 

imglist = dir(folder);
numoffile = size(imglist,1);
count = 0;
for i =3:1:numoffile
 if (strcmp(imglist(i).name,'Thumbs.db'))
     
     continue;
 end
 count = count+1;
 count
 namelist{count,1} = [imglist(i).name] ;
 imgfile = [folder, imglist(i).name] 
 [gbcolor, lccolor, compo,lines] = featureExtraction(imgfile);
 feature(count,1:7) = gbcolor(1,1:7);
 feature(count,8:18) = lccolor(1,1:11);
 feature(count,19:33) = compo(1,1:15);
 feature(count,34:37) = lines(1,1:4);
 
end
