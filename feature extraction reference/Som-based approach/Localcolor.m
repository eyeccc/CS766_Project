function [lc_t,lc_w,lc_contrast] = Localcolor(rgbimg)
height = size(rgbimg,1);
width = size(rgbimg,2);
[hueimg,segimg] = Hueimgseg(rgbimg,3);
%[segimg]=Graphcutsseg(rgbimg,3);
[y_0, x_0] = find(segimg==0);
[y_1, x_1] = find(segimg==1);
[y_2, x_2] = find(segimg==2);
colorMask_0 = zeros(height, width);
colorMask_1 = zeros(height, width);
colorMask_2 = zeros(height, width);
for i = 1:length(y_0)
    colorMask_0 (y_0(i),x_0(i))= 1;  
end
for i = 1:length(y_1)
     colorMask_1 (y_1(i),x_1(i))= 1;
end
for i= 1:length(y_2)
     colorMask_2 (y_2(i),x_2(i))= 1;
end
[hue,tempscore] = ColorTemperature(rgbimg);
lc_t(1,1) = sum(sum(tempscore.*colorMask_0))/length(y_0);
lc_t(1,2) = sum(sum(tempscore.*colorMask_1))/length(y_1);
lc_t(1,3) = sum(sum(tempscore.*colorMask_2))/length(y_2);
weightscore = ColorWeight(rgbimg);
lc_w(1,1) = sum(sum(weightscore.*colorMask_0))/length(y_0);
lc_w(1,2) = sum(sum(weightscore.*colorMask_1))/length(y_1);
lc_w(1,3) = sum(sum(weightscore.*colorMask_2))/length(y_2);
C = makecform('srgb2lab');
labimg = applycform((rgbimg),C);
C = makecform('lab2lch');
lchimg = applycform(lab2double(labimg),C);
luminance = lchimg(:,:,1);
saturation = lchimg(:,:,2);
hue = lchimg(:,:,3);
lum(1,1) = sum(sum(luminance.*colorMask_0))/length(y_0);
lum(1,2) = sum(sum(luminance.*colorMask_1))/length(y_1);
lum(1,3) = sum(sum(luminance.*colorMask_2))/length(y_2);

sat(1,1) = sum(sum(saturation.*colorMask_0))/length(y_0);
sat(1,2) = sum(sum(saturation.*colorMask_1))/length(y_1);
sat(1,3) = sum(sum(saturation.*colorMask_2))/length(y_2);

hu(1,1) = sum(sum(hue.*colorMask_0))/length(y_0);
hu(1,2) = sum(sum(hue.*colorMask_1))/length(y_1);
hu(1,3) = sum(sum(hue.*colorMask_2))/length(y_2);

lumdif = (sqrt(abs(lum(1,1).^2-lum(1,2).^2))+ sqrt(abs(lum(1,2).^2-lum(1,3).^2))+ sqrt(abs(lum(1,3).^2-lum(1,1).^2)))/3;
satdif = (sqrt(abs(sat(1,1).^2-sat(1,2).^2))+ sqrt(abs(sat(1,2).^2-sat(1,3).^2))+ sqrt(abs(sat(1,3).^2-sat(1,1).^2)))/3;
huedif = (sqrt(abs(hu(1,1).^2-hu(1,2).^2))+ sqrt(abs(hu(1,2).^2-hu(1,3).^2))+ sqrt(abs(hu(1,3).^2-hu(1,1).^2)))/3;
huetempdif = (sqrt(abs(lc_t(1,1).^2-lc_t(1,2).^2))+ sqrt(abs(lc_t(1,2).^2-lc_t(1,3).^2))+ sqrt(abs(lc_t(1,3).^2-lc_t(1,1).^2)))/3;
colorweightdif = (sqrt(abs(lc_w(1,1).^2-lc_w(1,2).^2))+ sqrt(abs(lc_w(1,2).^2-lc_w(1,3).^2))+ sqrt(abs(lc_w(1,3).^2-lc_w(1,1).^2)))/3;
lc_contrast(1,1)= lumdif/100;
lc_contrast(1,2)= satdif/100;
lc_contrast(1,3)= huedif/360;
lc_contrast(1,4)= huetempdif/1;
lc_contrast(1,5)= colorweightdif/1;
