function [saliency_img,format,sa_cv,sa_ecc, sa_re, focus,thirds] = Composition(rgbimg)
height = size(rgbimg,1);
width = size(rgbimg,2);
ratio = height/width;
golden_ratio = (1 +sqrt(5))/2;
format =ratio/(2*golden_ratio);

% map = gbvs(rgbimg);
% saliency_map = map.master_map_resized;
% Using simplified saliency map
 map = simpsal(rgbimg);
 saliency_map = mat2gray(imresize( map , [ height width ] )); 

[y, x] = find(saliency_map>=0.5); 
sa_cv = length(x)/(height*width);
salient_region = zeros(height, width);
 for i = 1:length(y)
     salient_region(y(i),x(i))=1;
 end
%   figure,
%   imshow(salient_region);
% calculate mass centre of salient region
 sumx = 0;
 sumy = 0;
 sumsaliency = 0;
%  sumx2 = 0;
%  sumy2 = 0;
 for i = 1:height
     for j = 1:width
         sumy = sumy+saliency_map(i,j)*i*salient_region(i,j);
         sumsaliency = sumsaliency + saliency_map(i,j)*salient_region(i,j);
         sumx = sumx+saliency_map(i,j)*j*salient_region(i,j);
%          sumy2 = sumy2+i*salient_region(i,j);
%          sumx2 = sumx2+j*salient_region(i,j);
     end
 end
mcenter(1,1) = round(sumy/sumsaliency);
mcenter(1,2) = round(sumx/sumsaliency);
% shapecenter(1,1)= round(sumy2/sum(sum(salient_region)));
% shapecenter(1,2)= round(sumx2/sum(sum(salient_region)));
[sa_ecc, sa_re] = ShapeRep(salient_region, mcenter);
[max_y,max_x]=find(saliency_map==max(max(saliency_map)));
focus(1,1) = max_y;
focus(1,2) = max_x;
saliency_img = heatmap_overlay(rgbimg,saliency_map.*salient_region);
% figure(1)
% imshow(saliency_img);
% hold on;
% a = plot(mcenter(1,2),mcenter(1,1),'g+','MarkerSize',8,'Linewidth',1.2);
% b = plot(focus(1,2),focus(1,1),'k*','MarkerSize',8,'Linewidth',1.2);
% % c = plot(shapecenter(1,2),shapecenter(1,1),'ro','MarkerSize',8,'Linewidth',1.2);
% legend([a,b],'Mass center of salient region','Focal point');
% hold off;

focus(1,1)=focus(1,1)/height;
focus(1,2)=focus(1,2)/width;
division = 1;
 hstep = 1:round(height/3):height;hstep(1,4)=height;
 wstep = 1:round(width/3):width;wstep(1,4)=width;
 for m = 1:3
     for n = 1:3
    thirds(1,division)= max(max(saliency_map(hstep(m):hstep(m+1),wstep(n):wstep(n+1))));
%     figure,
%     imshow(saliency_img(hstep(m):hstep(m+1),wstep(n):wstep(n+1)))
    division = division +1;
     end
 end
