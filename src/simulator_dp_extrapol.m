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

k_size = floor((1-(F./d)) * aperture_size);

tic;
for i = 1 : h % x
    for j = 1 : w %y
        ksizetb = k_size(i,j);

        % Move round inside extrapolation
        y1 = i - ksizetb / 2;
        y2 = i + ksizetb / 2;
        z1 = j;
        z2 = j + ksizetb / 2;
        
        if false
            if round(y1) == round(y2)
                y2 = y2 + 1;
            end

            if round(z1) == round(z2)
                z2 = z2 + 1;
            end
        end
        
        % Synthesizing Left Image      
        [img_v_list, count_v_list, y_int_list, z_int_list] = ...
          extrapolation(y1, z1, h, w, RGB_img(i, j, :), true);
        for idx = 1 : length(y_int_list)
            y_int = y_int_list{idx};
            z_int = z_int_list{idx};
            img_left(y_int, z_int, :) = img_left(y_int, z_int, :) + img_v_list{idx};
            count_left(y_int, z_int, :) = count_left(y_int, z_int, :) + count_v_list{idx};
        end
        
        [img_v_list, count_v_list, y_int_list, z_int_list] = ...
          extrapolation(y2, z1, h, w, RGB_img(i, j, :), false);
        for idx = 1 : length(y_int_list)
            y_int = y_int_list{idx};
            z_int = z_int_list{idx};
            img_left(y_int, z_int, :) = img_left(y_int, z_int, :) + img_v_list{idx};
            count_left(y_int, z_int, :) = count_left(y_int, z_int, :) + count_v_list{idx};
        end
        
        [img_v_list, count_v_list, y_int_list, z_int_list] = ...
          extrapolation(y1, z2, h, w, RGB_img(i, j, :), false);
        for idx = 1 : length(y_int_list)
            y_int = y_int_list{idx};
            z_int = z_int_list{idx};
            img_left(y_int, z_int, :) = img_left(y_int, z_int, :) + img_v_list{idx};
            count_left(y_int, z_int, :) = count_left(y_int, z_int, :) + count_v_list{idx};
        end
        
        [img_v_list, count_v_list, y_int_list, z_int_list] = ...
          extrapolation(y2, z2, h, w, RGB_img(i, j, :), true);
        for idx = 1 : length(y_int_list)
            y_int = y_int_list{idx};
            z_int = z_int_list{idx};
            img_left(y_int, z_int, :) = img_left(y_int, z_int, :) + img_v_list{idx};
            count_left(y_int, z_int, :) = count_left(y_int, z_int, :) + count_v_list{idx};
        end
        
        % Synthesizing Right Image
        z1 = j;
        z2 = j - ksizetb / 2;

        if false
            if z1 == z2 
                z2 = z2 - 1;
            end
        end
        
        [img_v_list, count_v_list, y_int_list, z_int_list] = ...
          extrapolation(y1, z1, h, w, RGB_img(i, j, :), true);
        for idx = 1 : length(y_int_list)
            y_int = y_int_list{idx};
            z_int = z_int_list{idx};
            img_right(y_int, z_int, :) = img_right(y_int, z_int, :) + img_v_list{idx};
            count_right(y_int, z_int, :) = count_right(y_int, z_int, :) + count_v_list{idx};
        end
        
        [img_v_list, count_v_list, y_int_list, z_int_list] = ...
          extrapolation(y2, z1, h, w, RGB_img(i, j, :), false);
        for idx = 1 : length(y_int_list)
            y_int = y_int_list{idx};
            z_int = z_int_list{idx};
            img_right(y_int, z_int, :) = img_right(y_int, z_int, :) + img_v_list{idx};
            count_right(y_int, z_int, :) = count_right(y_int, z_int, :) + count_v_list{idx};
        end
        
        [img_v_list, count_v_list, y_int_list, z_int_list] = ...
          extrapolation(y1, z2, h, w, RGB_img(i, j, :), false);
        for idx = 1 : length(y_int_list)
            y_int = y_int_list{idx};
            z_int = z_int_list{idx};
            img_right(y_int, z_int, :) = img_right(y_int, z_int, :) + img_v_list{idx};
            count_right(y_int, z_int, :) = count_right(y_int, z_int, :) + count_v_list{idx};
        end
        
        [img_v_list, count_v_list, y_int_list, z_int_list] = ...
          extrapolation(y2, z2, h, w, RGB_img(i, j, :), true);
        for idx = 1 : length(y_int_list)
            y_int = y_int_list{idx};
            z_int = z_int_list{idx};
            img_right(y_int, z_int, :) = img_right(y_int, z_int, :) + img_v_list{idx};
            count_right(y_int, z_int, :) = count_right(y_int, z_int, :) + count_v_list{idx};
        end
    end
end
toc

if false  % Zhiwei dump input and output data for CUDA test
    count_left_nonzero = single(count_left);
    count_left_nonzero(count_left == 0) = 1;
    img_left_avg = single(img_left ./ count_left_nonzero);

    count_right_nonzero = single(count_right);
    count_right_nonzero(count_right == 0) = 1;
    img_right_avg = single(img_right ./ count_right_nonzero);

    RGB_img_s = single(RGB_img);
    img_left_s = single(img_left);
    img_right_s = single(img_right);
    count_left_s = single(count_left);
    count_right_s = single(count_right);

    save('../data/matlab_dump.mat', 'RGB_img_s', 'img_left_s', ...
      'img_right_s', 'k_size', 'img_left_s', 'img_right_s', ...
      'count_left_s', 'count_right_s');
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
