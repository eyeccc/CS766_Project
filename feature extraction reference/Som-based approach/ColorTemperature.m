function [hue,tempscore] = ColorTemperature(rgbimg)
tempscore = zeros(size(rgbimg,1),size(rgbimg,2));
R = double(rgbimg(:,:,1)); 
G = double(rgbimg(:,:,2));
B = double(rgbimg(:,:,3));
b = 2*R-G-B;
a = sqrt(3)*(G-B);
[p,q] = find(b==0);
hue = (180/pi)*atan2(a,b);
for k = 1:length(p)
    hue(p(k),q(k))= Inf;
end

[i,j]=find(hue<0);
for k = 1:length(i)
    hue(i(k),j(k))= hue(i(k),j(k))+360;
end
for i=1:size(hue,1)
   
    for j = 1:size(hue,2)
    if hue(i,j)==Inf
    tempscore(i,j)=0;
    else
    tempscore(i,j) = HueTempScore(hue(i,j));
    end
    end
end