function [hueimg,segimg] = Hueimgseg(rgbimg,k)
%% hue based segmentation
%segimg is hueimg with colour numbers quantized to 3
rgbimg = imfilter(rgbimg,fspecial('gaussian',[5,5],2));
hsvimg = rgb2hsv(rgbimg);
hsvimg(:,:,2)=1;
hsvimg(:,:,3)=1;
hueimg = hsv2rgb(hsvimg);
% % for computing localized feature (maybe)
[segimg,map]= rgb2ind(hueimg,k,'nodither');
 % figure, imshow(segimg,map);