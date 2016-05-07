function[score] = HueTempScore(h)
%rotate hue for 60 degree
h = h + 60;
if h >= 360
    h=h-360;
end
% red
    if h>=0&&h<=75
        Hmax = 75; Hmin = 0;
        Wmax = 700; Wmin = 635;
    end
% orange
    if h>75&&h<=105
        Hmax = 105; Hmin = 75;
         Wmax = 635; Wmin = 590;
    end
% yellow
    if h>105&&h<=135
        Hmax = 135; Hmin = 105;
         Wmax = 590; Wmin = 560;
    end
% green
    if h>135&&h<=225
        Hmax = 225; Hmin = 135;
         Wmax = 560; Wmin = 490;
    end
% blue
    if h>225&&h<=315
        Hmax = 315; Hmin = 225;
         Wmax = 490; Wmin = 450;
    end  
% violet
    if h>315&&h<=360
        Hmax = 360; Hmin = 315;
         Wmax = 450; Wmin = 400;
    end
    
        


a = (Wmax-Wmin)/(Hmax-Hmin);
score = (Wmax-(a*(h-Hmin) )-400)/300;