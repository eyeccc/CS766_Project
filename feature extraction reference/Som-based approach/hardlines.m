function[line_para]=hardlines(rgbimg)
% smooth the image to reduce noise caused by canvas / brushstrokes
grayimg = rgb2gray(imfilter(rgbimg,fspecial('gaussian',[5,5],2)));
%grayimg = rgb2gray(rgbimg);
BW= edge(grayimg,'canny',[0,0.5]);
BW = bwmorph(BW,'dilate');
BW = bwmorph(BW,'thin');
% BW = edge(grayimg,'sobel',0.2);
%imshow(BW);
[H,T,R] = hough(BW,'RhoResolution',7,'ThetaResolution',5);
       P  = houghpeaks(H,100);
       x = T(P(:,2)); 
       y = R(P(:,1));
   %    plot(x,y,'s','color','white');
%   figure(1), imshow(BW), hold on
%        Find lines and plot them
       lines = houghlines(BW,T,R,P,'FillGap',2,'MinLength',11);
       
     
       distance = zeros(1,size(lines,2));
       slope = zeros(1,size(lines,2));
       hough_ratio = 0;
       meanslope = 0;
       stdslope = 0;
       meanlength = 0;
       if (~isempty(fieldnames(lines)))
       for k = 1:length(lines)
           xy = [lines(k).point1; lines(k).point2];
%           plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','red'); 
         distance(1,k)= sqrt((lines(k).point1(1,1)-lines(k).point2(1,1)).^2 + (lines(k).point1(1,1)-lines(k).point2(1,1)).^2);
         slope (1,k) = abs(lines(k).theta/90);   
       end
       area = size(BW,1)*size(BW,2);
       hough_ratio = sum(distance)/sum(sum(BW));
       meanlength = mean(distance)/sum(sum(BW));
       meanslope = mean(slope);
       stdslope = std(slope);
       end
   line_para(1,1)=hough_ratio;
   line_para(1,2)=meanslope;
   line_para(1,3)=stdslope;
   line_para(1,4)=meanlength;
   
   
      
