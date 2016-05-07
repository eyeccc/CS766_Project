function [gbcolor, lccolor, compo,lines] = featureExtraction(imgfile)
% this function calls all the feature extraction functions to extract the
% image features for one images. the extracted features includes global
% and local color, composition (based on visual saliency), and global line
% features
I = imread(imgfile);
height = size(I,1);
width = size(I,2);
aspect_ratio = width/height;
% if image size is too big shrink 1/2
th = 500;
if (height>th||width>th)
 I = imresize(I,[th,th*aspect_ratio]);
end
[glb_t,glb_w,glb_contrast] = Glbcolor(I);
gbcolor(1,1)=glb_t;
gbcolor(1,2)=glb_w;
gbcolor(1,3:7)= glb_contrast(1,1:5);

[lc_t,lc_w,lc_contrast] = Localcolor(I);
lccolor(1,1:3)= lc_t(1:3);
lccolor(1,4:6)= lc_w(1:3);
lccolor(1,7:11)=lc_contrast(1:5);


[saliency_img,format,sa_cv,sa_ecc, sa_re,focus,thirds] = Composition(I);
compo(1,1)= format;
compo(1,2)= sa_cv;
compo(1,3)= sa_ecc;
compo(1,4)= sa_re;
compo(1,5:6)= focus(1,1:2);
compo(1,7:15)= thirds(1,1:9);

[line_para] = hardlines(I);
lines(1,1:4)= line_para(1,1:4);


