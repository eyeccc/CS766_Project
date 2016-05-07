function [weightscore] = ColorWeight(rgbimg)

C = makecform('srgb2lab');
labimg = applycform((rgbimg),C);
C = makecform('lab2lch');
lchimg = applycform(lab2double(labimg),C);
lum = lchimg(:,:,1);
hue = lchimg(:,:,3);
[weightscore]=weightscorefun(hue,lum);