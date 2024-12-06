function [Reached_Pattern,Global_image, SNormlized_Maximum_intensity,num] = mergeimages(Maximum_intensity,O,EstimatedNumber,xx,yy,xpixel,ypixel,First_pattern,Size)
%% contact: leizhu201806@hotmail.com // version 07/2021
%%
Maximum_intensity = Maximum_intensity;% the maximum value of each image  r * r matrix
O = O;% the partial images, number = r;
EstimatedNumber = EstimatedNumber;% the estimated rank
xx = xx;% the position of maximum value of each image in X coordinate
yy = yy;% the position of maximum value of each image in Y coordinate
xpixel = xpixel;% the size of image in X coordinate
ypixel = ypixel;% the size of image in Y coordinate
% close all
Normlized_Maximum_intensity = zeros(size(Maximum_intensity));
% Size = 1500;
Global_image = zeros(Size,Size); %  Create the empty matrix
Central_Xposition = zeros(EstimatedNumber,1); %  Create the empty matrix to store the position of each emitter in X coordinate
Central_Yposition = zeros(EstimatedNumber,1); %  Create the empty matrix to store the position of each emitter in Y coordinate
ff = 0;
for k=1:EstimatedNumber
    if k == 1
        num = First_pattern % number of the pattern, same as k in the deconvolution procedure
        SNormlized_Maximum_intensity = Maximum_intensity(:,num)/max(Maximum_intensity(:,num));
        [aa,indices] = sort(SNormlized_Maximum_intensity,'descend'); % aa is the sorted matrix and indices are the Indices of Sorted Elements
        Known_Finger = indices; % 
        Reached_Pattern(k) = Known_Finger(1); % Reached_pattern (K) store the index of largest element of the SNormlized_Maximum_intensity
        for zz = 1:1
            % calculate the relative postion
            global_image = ones(size(Global_image)); % 500x500 for example
            Central_Xposition(Reached_Pattern(k)) = floor(Size/2)-1;
            Central_Yposition(Reached_Pattern(k)) = floor(Size/2)-1;
            hh = histogram(O{Reached_Pattern(k)},100);% bin 100
            zz = hh.Values;
            background = hh.BinEdges(find(zz == max(zz))+1);
            global_image = global_image*background;
            global_image(Central_Xposition(Reached_Pattern(k))+(-floor(xpixel/2):floor(xpixel/2)-1),...
                Central_Yposition(Reached_Pattern(k))+(-floor(ypixel/2):floor(ypixel/2)-1))...
                = O{Reached_Pattern(k)};
            Global_image = Global_image + global_image; % merge them into the blobal image.
            ff = ff+1;
        end
%         figure
%         imshow(Global_image,[]);colormap hot;
    else % k = 2~r from here
        for zz = 2:(EstimatedNumber)
            NextPSF = indices(zz);
            check = ismember(Reached_Pattern,NextPSF);
            disp(NextPSF);
            disp('------------------------------')
           
            if check == 0
                Reached_Pattern(k) = NextPSF;
                break
            elseif check == 1
            end
        end
        if SNormlized_Maximum_intensity(NextPSF)<= 0.15
            Reached_Pattern(k) = [];
            break
        end
        global_image = ones(size(Global_image));
        temp = Reached_Pattern(k-1);
        Central_Xposition(Reached_Pattern(k)) = Central_Xposition(temp)-...
            0*floor(Size/2)+xx(Reached_Pattern(k),temp)-xx(temp,temp);
        Central_Yposition(Reached_Pattern(k)) = Central_Yposition(temp)-...
            0*floor(Size/2)+yy(Reached_Pattern(k),temp)-yy(temp,temp);
        hh = histogram(O{Reached_Pattern(k)},100);
        zz = hh.Values;
        background = hh.BinEdges(find(zz == max(zz))+1);
        global_image = global_image*background;
        global_image((Central_Xposition(Reached_Pattern(k))-floor(xpixel/2)):(Central_Xposition(Reached_Pattern(k))+floor(xpixel/2)-1),...
            (Central_Yposition(Reached_Pattern(k))-floor(ypixel/2)):(Central_Yposition(Reached_Pattern(k))+floor(ypixel/2))-1)...
            = O{Reached_Pattern(k)};
        Global_image = Global_image + global_image; % merge them into the blobal image.
        ff = ff+1;
        imshow(Global_image,[]);colormap hot;
        SNormlized_Maximum_intensity = Maximum_intensity(:,Reached_Pattern(k))/max(Maximum_intensity(:,Reached_Pattern(k)));
        [aa,indices] = sort(SNormlized_Maximum_intensity,'descend');
       % disp(Central_Xposition(Reached_Pattern(k)));
        % disp(Reached_Pattern(k));
    end
end
