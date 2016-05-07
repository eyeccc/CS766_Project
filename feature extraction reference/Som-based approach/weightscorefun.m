function[ws] = weightscorefun(hue, lum)
lum = lum/100;
hue = (hue*pi)/180;
fh = -0.03312*hue.^2 + 0.3331*hue -0.03369;
  a = -6.327e-013 ; 
  b =  27.13  ;
  c =  1.016  ;
  d =  -0.968 ; 
fl =a*exp(b*lum) + c*exp(d*lum);
ws = abs(0.1*fh + 0.9*fl);
if (fl==0&fh==0)
    ws = 1;
end
if (fl==0&fh==100)
    ws= 0;
end
