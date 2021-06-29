
%% camera para
para = 35;
% m -> mm
f=para/1e3; 
% focal length in number of pixels raise raise
f_rgb=3000;   
pixelsize=f/f_rgb;
% aperture size in grid of pixels
aperture_size=floor(f_rgb/5);   
Fd = 7;
% sacle for depth range
scaled = 1.0; 
% crop image boundary
crop = 20; 

%% load data
img_name = imread('../data/input.png');
RGB_img = im2double(img_name);

depth_name = load('../data/depths.mat');
depth_in = depth_name.depths/scaled;

% depth in number of pixels
depth_pixel = depth_in/(pixelsize);

% d'
d = (f_rgb*depth_pixel)./(depth_pixel-f_rgb); 
% focal distance
F = min(d(:))-Fd; 
% disparity
disp = ((d-F)./d).*aperture_size/2; 

%%
[h,w,~]=size(depth_pixel);

img_left = zeros(size(RGB_img));
count_left = zeros(size(d));

img_right = zeros(size(RGB_img));
count_right = zeros(size(d));

k_size = floor( (1-(F./d) ) * aperture_size );

for i = 1:h % x
    for j = 1 :w %y
        
        ksizetb = k_size(i,j);

        y1 = i - floor(ksizetb/2);
        y2 = i + floor(ksizetb/2);
        z1 = j;
        z2 = j +floor(ksizetb/2);
        
        if y1==y2 
            y2=y2+1;
        end
        if z1==z2 
            z2=z2+1;
        end
        
        
        y1 = max(1,min(y1,h));
        y2 = max(1,min(y2,h));
        
        z1 = max(1,min(z1,w));
        z2 = max(1,min(z2,w));
        
        % Synthesizing Left Image
        img_left(y1,z1,:)=img_left(y1,z1,:)+RGB_img(i,j,:);
        img_left(y2,z1,:)=img_left(y2,z1,:)-RGB_img(i,j,:);
        img_left(y1,z2,:)=img_left(y1,z2,:)-RGB_img(i,j,:);
        img_left(y2,z2,:)=img_left(y2,z2,:)+RGB_img(i,j,:);
        
        count_left(y1,z1)= count_left(y1,z1)+1;
        count_left(y2,z1)= count_left(y2,z1)-1;
        count_left(y1,z2)= count_left(y1,z2)-1;
        count_left(y2,z2)= count_left(y2,z2)+1;
        
        % Synthesizing Right Image
        z1 = j;
        z2 = j - floor(ksizetb/2);

        if z1==z2 
            z2=z2-1;
        end
        
        z1 = max(1,min(z1,w));
        z2 = max(1,min(z2,w));
        
        img_right(y1,z1,:)=img_right(y1,z1,:)+RGB_img(i,j,:);
        img_right(y2,z1,:)=img_right(y2,z1,:)-RGB_img(i,j,:);
        img_right(y1,z2,:)=img_right(y1,z2,:)-RGB_img(i,j,:);
        img_right(y2,z2,:)=img_right(y2,z2,:)+RGB_img(i,j,:);
        
        count_right(y1,z1)= count_right(y1,z1)+1;
        count_right(y2,z1)= count_right(y2,z1)-1;
        count_right(y1,z2)= count_right(y1,z2)-1;
        count_right(y2,z2)= count_right(y2,z2)+1;
        
    end
end
%%
% Intigral image - left
integral_image=(integralImage(img_left));
integral_count=(integralImage(count_left));
integral_count(integral_count==0) = 1;

integral_count(integral_count==0) = 1;
img_left=(integral_image(1:end-1,1:end-1,:))./(integral_count(1:end-1,1:end-1,:));

% Intigral image - right
integral_image=(integralImage(img_right));
integral_count=(integralImage(count_right));
integral_count(integral_count==0) = -1;

integral_count(integral_count==0) = 1;
img_right=(integral_image(1:end-1,1:end-1,:))./(integral_count(1:end-1,1:end-1,:));

%% avoid hole in the image - post process
% [X,Y] = meshgrid(1:w,1:h);
% vXr = X + double(disp);
% midiml = RGB_img;
% midimr = RGB_img;
% vXl = X - double(disp);
% 
% for c = 1:3
%     iml = img_left(:,:,c);
%     imr = img_right(:,:,c);
%     midimr(:,:,c) = griddata(X,Y,iml,vXr,Y);
%     midiml(:,:,c) = griddata(X,Y,imr,vXl,Y);
% end
% img_left(idxl) = midiml(idxl);
% img_right(idxr) = midimr(idxr);


img_left = imresize(img_left(crop:end-crop,crop:end-crop,:), [480,640]);
img_right = imresize(img_right(crop:end-crop,crop:end-crop,:), [480,640]);
RGB_img = imresize(RGB_img(crop:end-crop,crop:end-crop,:), [480,640]);

idx = k_size<=2;
idx = repmat(idx,[1,1,3]);
img_left(idx)=RGB_img(idx);
img_right(idx)=RGB_img(idx);

imwrite(img_left, '../results/oult_l.png');
imwrite(img_right, '../results/oult_r.png');
imwrite(RGB_img, '../results/oult_gt.png');