%
% Zhiwei Xu <zhiwei.xu@anu.edu.au>
%

%% load data
data_dir = '../../data/single';
max_index = 1;
save_dir = fullfile(data_dir, 'matlab');

if ~exist(save_dir, 'dir')
  mkdir(save_dir);
end

for index = 1 : 1 : max_index
  fprintf('Processing %02d...\n', index);
  processing_single(data_dir, index);
end

function processing_single(data_dir, index)
  %% camera para
  para = 35;
  % m -> mm
  f = para/1e3;
  % focal length in number of pixels raise raise
  f_rgb = 3000;
  pixelsize = f/f_rgb;
  % aperture size in grid of pixels
  aperture_size = floor(f_rgb/5);
  Fd = 7;
  % sacle for depth range
  scaled = 1.0;
  % crop image boundary
  crop = 20;

  img_path = fullfile(data_dir, sprintf('input_%02d.png', index));
  depth_path = fullfile(data_dir, sprintf('depth_%02d.mat', index));
  save_path = fullfile(data_dir, sprintf('matlab/matlab_%02d.mat', index));

  if ~exist(img_path, 'file') || ~exist(depth_path, 'file') ...
      || exist(save_path, 'file')
    return;
  end

  img_name = imread(img_path);
  depth_name = load(depth_path);
  RGB_img = im2double(img_name);

  if isfield(depth_name, 'disp')
    disp = depth_name.disp;
    k_size = disp;
  else
    depth_in = depth_name.depths / scaled;

    % depth in number of pixels
    depth_pixel = depth_in / pixelsize;

    % d'
    d = (f_rgb * depth_pixel) ./ (depth_pixel - f_rgb);
    % focal distance
    F = min(d(:)) - Fd;
    % disparity
    disp = ((d - F) ./ d) .* aperture_size / 2;
    k_size = floor((1 - (F ./ d)) * aperture_size);
  end

  %%
  [h, w, c] = size(img_name);  %-----------------

  img_left = zeros(size(RGB_img));
  count_left = zeros(size(disp));

  img_right = zeros(size(RGB_img));
  count_right = zeros(size(disp));

  tic;
  RGB_img = single(RGB_img);
  k_size = single(k_size);
  img_left = single(img_left);
  img_right = single(img_right);
  count_left = single(count_left);
  count_right = single(count_right);

  for i = 1 : h % x
      for j = 1 : w %y
          ksizetb = k_size(i, j);

          % Move round inside extrapolation
          y1 = i; %-----------------
          y2 = i + ksizetb; %-----------------
          z1 = j;
          z2 = j + ksizetb; %-----------------

          if false
              if floor(y1) == floor(y2)
                  y2 = y2 + 1;
              end

              if floor(z1) == floor(z2)
                  z2 = z2 + 1;
              end
          end

          % Synthesizing Left Image      
          [img_v_list, count_v_list, y_int_list, z_int_list] = ...
            extrapolation(y1, z1, h, w, RGB_img(i, j, :));
          for idx = 1 : length(y_int_list)
              y_int = y_int_list{idx};
              z_int = z_int_list{idx};
              img_left(y_int, z_int, :) = img_left(y_int, z_int, :) + img_v_list{idx};
              count_left(y_int, z_int, :) = count_left(y_int, z_int, :) + count_v_list{idx};
          end

          [img_v_list, count_v_list, y_int_list, z_int_list] = ...
            extrapolation(y2, z1, h, w, RGB_img(i, j, :));
          for idx = 1 : length(y_int_list)
              y_int = y_int_list{idx};
              z_int = z_int_list{idx};
              img_left(y_int, z_int, :) = img_left(y_int, z_int, :) - img_v_list{idx};
              count_left(y_int, z_int, :) = count_left(y_int, z_int, :) - count_v_list{idx};
          end

          [img_v_list, count_v_list, y_int_list, z_int_list] = ...
            extrapolation(y1, z2, h, w, RGB_img(i, j, :));
          for idx = 1 : length(y_int_list)
              y_int = y_int_list{idx};
              z_int = z_int_list{idx};
              img_left(y_int, z_int, :) = img_left(y_int, z_int, :) - img_v_list{idx};
              count_left(y_int, z_int, :) = count_left(y_int, z_int, :) - count_v_list{idx};
          end

          [img_v_list, count_v_list, y_int_list, z_int_list] = ...
            extrapolation(y2, z2, h, w, RGB_img(i, j, :));
          for idx = 1 : length(y_int_list)
              y_int = y_int_list{idx};
              z_int = z_int_list{idx};
              img_left(y_int, z_int, :) = img_left(y_int, z_int, :) + img_v_list{idx};
              count_left(y_int, z_int, :) = count_left(y_int, z_int, :) + count_v_list{idx};
          end

          % Synthesizing Right Image
          z1 = j - ksizetb;   %-----------------
          z2 = j;   %-----------------

          if false
              if floor(z1) == floor(z2)
                  z1 = z1 - 1;   %----------------- 
              end
          end

          [img_v_list, count_v_list, y_int_list, z_int_list] = ...
            extrapolation(y1, z1, h, w, RGB_img(i, j, :));
          for idx = 1 : length(y_int_list)
              y_int = y_int_list{idx};
              z_int = z_int_list{idx};
              img_right(y_int, z_int, :) = img_right(y_int, z_int, :) + img_v_list{idx};
              count_right(y_int, z_int, :) = count_right(y_int, z_int, :) + count_v_list{idx};
          end

          [img_v_list, count_v_list, y_int_list, z_int_list] = ...
            extrapolation(y2, z1, h, w, RGB_img(i, j, :));
          for idx = 1 : length(y_int_list)
              y_int = y_int_list{idx};
              z_int = z_int_list{idx};
              img_right(y_int, z_int, :) = img_right(y_int, z_int, :) - img_v_list{idx};
              count_right(y_int, z_int, :) = count_right(y_int, z_int, :) - count_v_list{idx};
          end

          [img_v_list, count_v_list, y_int_list, z_int_list] = ...
            extrapolation(y1, z2, h, w, RGB_img(i, j, :));
          for idx = 1 : length(y_int_list)
              y_int = y_int_list{idx};
              z_int = z_int_list{idx};
              img_right(y_int, z_int, :) = img_right(y_int, z_int, :) - img_v_list{idx};
              count_right(y_int, z_int, :) = count_right(y_int, z_int, :) - count_v_list{idx};
          end

          [img_v_list, count_v_list, y_int_list, z_int_list] = ...
            extrapolation(y2, z2, h, w, RGB_img(i, j, :));
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

      save(fullfile(data_dir, sprintf('matlab/matlab_dump_%02d.mat', index)), ...
        'RGB_img_s', 'k_size', 'img_left_s', 'img_right_s', ...
        'count_left_s', 'count_right_s');
  end

  %%
  % Intigral image - left
  integral_image = (integralImage(img_left));
  integral_count = (integralImage(count_left));
  integral_count(integral_count == 0) = 1;

  integral_count = repmat(integral_count, [1, 1, c]); %----------------- 
  img_left_final = integral_image ./ integral_count; %----------------- 
  img_left_final = (img_left_final(2:end, 2:end, :)); %----------------- 

  % Intigral image - right
  integral_image = (integralImage(img_right));
  integral_count = (integralImage(count_right));
  integral_count(integral_count == 0) = 1; %----------------- 

  integral_count = repmat(integral_count,[1,1,c]); %----------------- 
  img_right_final = integral_image ./ integral_count; %----------------- 
  img_right_final = (img_right_final(2:end, 2:end, :)); %----------------- 

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

  if false
    img_left_final = imresize(img_left_final(crop:end-crop,crop:end-crop,:), [480,640]);
    img_right_final = imresize(img_right_final(crop:end-crop,crop:end-crop,:), [480,640]);
    RGB_img_final = imresize(RGB_img(crop:end-crop,crop:end-crop,:), [480,640]);
  else
    RGB_img_final = RGB_img;
  end

  idx = k_size<=2;
  idx = repmat(idx,[1,1,3]);
  img_left_final(idx) = RGB_img_final(idx);
  img_right_final(idx) = RGB_img_final(idx);

  imwrite(img_left_final, fullfile(data_dir, sprintf('matlab/left_syn_matlab_%02d.png', index)));
  imwrite(img_right_final, fullfile(data_dir, sprintf('matlab/right_syn_matlab_%02d.png', index)));
  imwrite(RGB_img_final, fullfile(data_dir, sprintf('matlab/gt_matlab_%02d.png', index)));

  save(save_path, 'img_left', 'img_right', ...
    'count_left', 'count_right', 'RGB_img', 'k_size', ...
    'img_left_final', 'img_right_final', 'RGB_img_final');
end
