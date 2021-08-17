function [img_list, count_list, y_int_list, z_int_list] = ...
  extrapolation(y, z, h, w, src_pixel)
    img_list = {};
    count_list = {};
    y_int_list = {};
    z_int_list = {};
    
    % This is to assign to the nearest neighbour
    % if enable this, in the loops below, should have a condition weight==1
    if false
      y = floor(y);
      z = floor(z);
      y_int_loop = unique([floor(y)]);
      z_int_loop = unique([floor(z)]);
    else
      y_int_loop = unique([floor(y), ceil(y)]);
      z_int_loop = unique([floor(z), ceil(z)]);
    end
    
    for i = 1 : length(y_int_loop)
        for j = 1 : length(z_int_loop)
            y_int = y_int_loop(i);
            z_int = z_int_loop(j);
            y_weight = 1 - abs(y_int - y);
            z_weight = 1 - abs(z_int - z);
            weight = y_weight * z_weight;
            
            y_int = max(1, min(y_int, h));
            z_int = max(1, min(z_int, w));

            img_v = weight * src_pixel;
            count_v = weight;

            img_list{end+1} = img_v;
            count_list{end+1} = count_v;
            y_int_list{end+1} = y_int;
            z_int_list{end+1} = z_int;
        end
    end
end
