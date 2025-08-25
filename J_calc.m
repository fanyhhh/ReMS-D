function [row_temp] = J_calc(I_locolmean_warped_same_area_cut, mask_locolmean_warped_same_area_cut, Rate_locolmean_warped_same_area_cut, A0,channel_g0,channel_g_rate,channel_b0,channel_b_rate,WB_type,flag_show,solver_type,python_mode,calc_mode)
points_part=[];
for img_num=1:length(I_locolmean_warped_same_area_cut)
    switch WB_type
        case 'no cali'
            points_all(:,:,:,img_num)=I_locolmean_warped_same_area_cut{img_num}./mask_locolmean_warped_same_area_cut{img_num};
        case 'cali I/M'
            points_all(:,:,1,img_num)=I_locolmean_warped_same_area_cut{img_num}(:,:,1)./mask_locolmean_warped_same_area_cut{img_num}(:,:,1);
            points_all(:,:,2,img_num)=(I_locolmean_warped_same_area_cut{img_num}(:,:,2)./mask_locolmean_warped_same_area_cut{img_num}(:,:,2) - channel_g0(img_num)) / channel_g_rate(img_num);
            points_all(:,:,3,img_num)=(I_locolmean_warped_same_area_cut{img_num}(:,:,3)./mask_locolmean_warped_same_area_cut{img_num}(:,:,3) - channel_b0(img_num)) / channel_b_rate(img_num);
        case 'cali I_mean'
            points_all(:,:,1,img_num)=I_locolmean_warped_same_area_cut{img_num}(:,:,1)./mask_locolmean_warped_same_area_cut{img_num}(:,:,1);
            points_all(:,:,2,img_num)=((I_locolmean_warped_same_area_cut{img_num}(:,:,2) - channel_g0(img_num)) / channel_g_rate(img_num))./mask_locolmean_warped_same_area_cut{img_num}(:,:,2);
            points_all(:,:,3,img_num)=((I_locolmean_warped_same_area_cut{img_num}(:,:,3) - channel_b0(img_num)) / channel_b_rate(img_num))./mask_locolmean_warped_same_area_cut{img_num}(:,:,3);
    end
end

points_part=imresize(points_all,0.1);
points_part = permute(points_part, [4, 3, 1, 2]);
old_size = size(points_part);
new_size = [old_size(1:2), prod(old_size(3:end))];
points_part = reshape(points_part, new_size);

J_rate_part=[];

Patch_size_GWA=150;
Kernel=ones(Patch_size_GWA,Patch_size_GWA);
for img_num=1:length(I_locolmean_warped_same_area_cut)
    J_rate_all(:,:,:,img_num)=Rate_locolmean_warped_same_area_cut{img_num}./mask_locolmean_warped_same_area_cut{img_num};
end

J_rate_part=imresize(J_rate_all,0.1);
J_rate_part = permute(J_rate_part, [4, 3, 1, 2]);
old_size = size(J_rate_part);
new_size = [old_size(1:2), prod(old_size(3:end))];
J_rate_part = reshape(J_rate_part, new_size);

J0_part = rand([1,3,size(J_rate_part,3)]);

V_part=points_part-repmat(A0,[size(points_part,1),1,size(points_part,3)]);
V_part=V_part ./ repmat(sqrt(V_part(:,1,:).^2+V_part(:,2,:).^2+V_part(:,3,:).^2),[1,3,1]);
switch solver_type
    case 'python'
        py_envi='C:/Users/admin/anaconda3/envs/opt_func/python.exe';

        currentDateTime = datetime('now', 'Format', 'yyyyMMdd_HHmmssSSS');
        currentTimestamp = char(currentDateTime);
        mkdir(currentTimestamp);
        save([currentTimestamp,'\opt_data_temp.mat'], 'A0', 'J_rate_part', 'V_part', 'points_part');

        switch python_mode
            case 'separate'
                if calc_mode == 'cpu'
                    sourceFile = 'my_opt_func_separate.py';
                    targetFile = fullfile([currentTimestamp, '\my_opt_func_separate.py']);
                elseif calc_mode == 'gpu'
                    sourceFile = 'my_opt_func_separate_gpu.py';
                    targetFile = fullfile([currentTimestamp, '\my_opt_func_separate_gpu.py']);
                end
                copyfile(sourceFile, targetFile);

                cd(currentTimestamp)
                if ~exist('opted_data_temp.mat', 'file')
                    if calc_mode == 'cpu'
                        system([py_envi,' my_opt_func_separate.py']);
                    elseif calc_mode == 'gpu'
                        system(['conda run -n drone python my_opt_func_separate_gpu.py']);
                    end

                end
                load('opted_data_temp.mat');
                cd ..
            
        end

        J_result_part=J0_result;
end
row_result_part=J_result_part./repmat(A0,[1,1,size(J_result_part,3)]);

J_result_part_shape=permute(J_result_part, [3, 1, 2]);
J_result_part_shape=reshape(J_result_part_shape, [old_size(3:end),3]);
J_result_part_all=imresize(J_result_part_shape,[size(I_locolmean_warped_same_area_cut{1},[1,2])]);

row_result_part_shape=J_result_part_shape./repmat(permute(A0,[1,3,2]),[size(J_result_part_shape,1,2),1]);
row_result_part_all=imresize(row_result_part_shape,[size(I_locolmean_warped_same_area_cut{1},[1,2])]);

Patch_size_GWA=150;
Kernel=ones(Patch_size_GWA,Patch_size_GWA);%DE=sourcePic(:,:,3);
for i=1:3
    row_result_locolmean(:,:,i)=imfilter(row_result_part_all(:,:,i), Kernel,'symmetric')/sum(sum(Kernel));
end

for img_num=1:length(I_locolmean_warped_same_area_cut)
    row_temp{img_num}=row_result_locolmean;
end

d_A0_I=sum((repmat(A0,[size(points_part,1),1,size(points_part,3)])-points_part).^2,2);
d_A0_J=sum((repmat(A0,[size(points_part,1),1,size(points_part,3)])-J_result_part .* J_rate_part).^2,2);
t_result_part=sqrt(d_A0_I./d_A0_J);

t_result_shape=permute(t_result_part, [3, 1, 2]);
t_result_shape=reshape(t_result_shape, [old_size(3:end),length(I_locolmean_warped_same_area_cut)]);
t_result_shape_all=imresize(t_result_shape,[size(I_locolmean_warped_same_area_cut{1},[1,2])]);


for img_num=1:length(I_locolmean_warped_same_area_cut)
    t_result_locolmean(:,:,img_num)=imfilter(t_result_shape_all(:,:,img_num), Kernel,'symmetric')/sum(sum(Kernel));
end

for img_num=1:length(I_locolmean_warped_same_area_cut)
    t_temp{img_num}=t_result_shape(:,:,img_num);
    %     t_temp{img_num}=t_result_locolmean(:,:,img_num);
end

J0_all = repmat(J_result_part_all,[1,1,1,size(points_all,4)]) .* J_rate_all;
V=points_all-repmat(permute(A0,[3,1,2]),[size(points_all,[1,2]),1,size(points_all,4)]);  % 方向向量
cross0_all=cross((J0_all - repmat(permute(A0,[3,1,2]),[size(points_all,[1,2]),1,size(points_all,4)])), V);

distance_all = sqrt(cross0_all(:,:,1,:).^2 + cross0_all(:,:,2,:).^2 + cross0_all(:,:,3,:).^2);

d_A0_I=sum((repmat(permute(A0,[3,1,2]),[size(points_all,[1,2]),1,size(points_all,4)])-points_all).^2,3);
d_A0_J=sum((repmat(permute(A0,[3,1,2]),[size(points_all,[1,2]),1,size(points_all,4)])-J0_all).^2,3);
t_result=sqrt(d_A0_I./d_A0_J);
