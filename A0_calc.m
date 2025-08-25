function [A0] = A0_calc(I_locolmean_warped_same_area_cut, mask_warped_same_area_cut,A0_min,channel_g0,channel_g_rate,channel_b0,channel_b_rate,WB_type,solver_type)

for img_num=1:length(I_locolmean_warped_same_area_cut)
    switch WB_type
        case 'no cali'
            points(:,:,:,img_num)=I_locolmean_warped_same_area_cut{img_num}./mask_warped_same_area_cut{img_num};
        case 'cali I/M'
            points(:,:,1,img_num)=I_locolmean_warped_same_area_cut{img_num}(:,:,1)./mask_warped_same_area_cut{img_num}(:,:,1);
            points(:,:,2,img_num)=(I_locolmean_warped_same_area_cut{img_num}(:,:,2)./mask_warped_same_area_cut{img_num}(:,:,2) - channel_g0(img_num)) / channel_g_rate(img_num);
            points(:,:,3,img_num)=(I_locolmean_warped_same_area_cut{img_num}(:,:,3)./mask_warped_same_area_cut{img_num}(:,:,3) - channel_b0(img_num)) / channel_b_rate(img_num);
        case 'cali I_mean'
            points(:,:,1,img_num)=I_locolmean_warped_same_area_cut{img_num}(:,:,1)./mask_warped_same_area_cut{img_num}(:,:,1);
            points(:,:,2,img_num)=((I_locolmean_warped_same_area_cut{img_num}(:,:,2) - channel_g0(img_num)) / channel_g_rate(img_num))./mask_warped_same_area_cut{img_num}(:,:,2);
            points(:,:,3,img_num)=((I_locolmean_warped_same_area_cut{img_num}(:,:,3) - channel_b0(img_num)) / channel_b_rate(img_num))./mask_warped_same_area_cut{img_num}(:,:,3);
    end
end
points = permute(points, [4, 3, 1, 2]);

old_size = size(points);
new_size = [old_size(1:2), prod(old_size(3:end))];
points = reshape(points, new_size);

data_matrix = [points(:,1,:), points(:,2,:), ones(size(points, 1), 1, size(points,3))];
target_vector = -points(:,3,:);
for i=1:size(data_matrix,3)
    coefficients(:,:,i) = data_matrix(:,:,i) \ target_vector(:,:,i);
end
A = squeeze(coefficients(1,:,:));
B = squeeze(coefficients(2,:,:));
C = ones(size(A));
D = squeeze(coefficients(3,:,:));

distance_norm=sqrt(A.^2+B.^2+C.^2);
A=A./distance_norm;
B=B./distance_norm;
C=C./distance_norm;
D=D./distance_norm;

A0_init = rand(1,3);
target_func = @(xyz) cost_A0(xyz, A, B, C, D);

switch solver_type
    case 'python'
        pyversion

        mymodule = py.importlib.import_module('my_opt_func_A0');
        py.importlib.reload(mymodule);

        A0_result = mymodule.my_opt_func_A0(py.numpy.array(A0_init), py.numpy.array(A0_min), py.numpy.array([1 1 1]), ...
            py.numpy.array(A), py.numpy.array(B), py.numpy.array(C), py.numpy.array(D));
        A0 = double(A0_result);
end

end