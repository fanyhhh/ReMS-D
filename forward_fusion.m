function [img_fusion_rate] = forward_fusion(I_locolmean_warped_same_area_cut,Rate_warped_same_area_cut,img_warped_same_area_cut,mask_warped_same_area_cut,J0_all,A0)

img_num_all=length(J0_all);

ref=round(length(J0_all)/2);
img_haze = J0_all{ref};

points_part=[];
for img_num=1:img_num_all
    points_all(:,:,:,img_num)=img_warped_same_area_cut{img_num}./mask_warped_same_area_cut{img_num};
end

points_part=points_all;

points_part = permute(points_part, [4, 3, 1, 2]);

old_size = size(points_part);
new_size = [old_size(1:2), prod(old_size(3:end))];
points_part = reshape(points_part, new_size);

J_rate_part=[];
Patch_size_GWA=150;
Kernel=ones(Patch_size_GWA,Patch_size_GWA);%DE=sourcePic(:,:,3);
for img_num=1:img_num_all
    J_rate_all(:,:,:,img_num)=Rate_warped_same_area_cut{img_num}./mask_warped_same_area_cut{img_num};
end

J_rate_part = permute(J_rate_all, [4, 3, 1, 2]);

old_size = size(J_rate_part);
new_size = [old_size(1:2), prod(old_size(3:end))];
J_rate_part = reshape(J_rate_part, new_size);
disp(size(J_rate_part));

J0_part=[];
for img_num=1:img_num_all
    J0_part(:,:,:,img_num)=J0_all{img_num};
end
J0_part = permute(J0_part, [4, 3, 1, 2]);

old_size = size(J0_part);
new_size = [old_size(1:2), prod(old_size(3:end))];
J0_part = reshape(J0_part, new_size);

J0_J_rate_part=J0_part .* J_rate_part;

V=points_part-repmat(A0,[size(points_part,1),1,size(points_part,3)]);
cross0_part=cross((J0_J_rate_part - repmat(A0,[size(points_part,1),1,size(points_part,3)])), V);
distance_part = sqrt(cross0_part(:,1,:).^2 + cross0_part(:,2,:).^2 + cross0_part(:,3,:).^2);

d_A0_I=sum((repmat(A0,[size(points_part,1),1,size(points_part,3)])-points_part).^2,2);
d_A0_J=sum((repmat(A0,[size(points_part,1),1,size(points_part,3)])-J0_J_rate_part).^2,2);
t_result=sqrt(d_A0_I./d_A0_J);

for img_num=1:length(J0_all)
    mean_rgb = mean(J0_all{img_num},[1,2]);
    J0_all{img_num} = J0_all{img_num} .* mean(mean_rgb) ./ mean_rgb;
end

J0_part=[];
for img_num=1:img_num_all
    J0_part(:,:,:,img_num)=J0_all{img_num};
end

J0_part = permute(J0_part, [4, 3, 1, 2]);

old_size = size(J0_part);
new_size = [old_size(1:2), prod(old_size(3:end))];
J0_part = reshape(J0_part, new_size);

w1 = t_result.^4 ./ sum(t_result.^4,1);
J_fusion_part1 = sum(J0_part .* w1,1);

w2 = distance_part ./ sum(distance_part,1);
w2 = exp(-w2/0.1);
w2 = w2 ./ sum(w2,1);
J_fusion_part2 = sum(J0_part .* w2,1);

t_mask = ones(size(t_result));
[t_sorted, index] = sort(t_result,1,'descend');

for img_num=1:length(J0_all)
    for out_num=length(J0_all) -1 : length(J0_all)
        [row,col] = find(index(out_num,:,:) == img_num);
        index_line = img_num + (col-1) * length(J0_all);
        t_mask(index_line) = 0;
    end
end

w3 = distance_part.*t_mask ./ sum(distance_part.*t_mask,1);
w3 = exp(-w3/0.1);
w3 = w3 .*t_mask;
w3 = w3 ./ sum(w3,1);
J_fusion_part3 = sum(J0_part .* w3,1);

num_point=791098;
disp([w1(:,:,num_point)'; w2(:,:,num_point)'; w3(:,:,num_point)']);

J_fusion_part_shape1=permute(J_fusion_part1, [3, 1, 2]);
J_fusion_part_shape1=reshape(J_fusion_part_shape1, [old_size(3:end),3]);
J_fusion_part_all1=imresize(J_fusion_part_shape1,[size(I_locolmean_warped_same_area_cut{1},[1,2])]);

J_fusion_part_shape2=permute(J_fusion_part2, [3, 1, 2]);
J_fusion_part_shape2=reshape(J_fusion_part_shape2, [old_size(3:end),3]);
J_fusion_part_all2=imresize(J_fusion_part_shape2,[size(I_locolmean_warped_same_area_cut{1},[1,2])]);

J_fusion_part_shape3=permute(J_fusion_part3, [3, 1, 2]);
J_fusion_part_shape3=reshape(J_fusion_part_shape3, [old_size(3:end),3]);
J_fusion_part_all3=imresize(J_fusion_part_shape3,[size(I_locolmean_warped_same_area_cut{1},[1,2])]);

img_fusion=J_fusion_part_all3;
rate = 0.25/mean(img_fusion(:));

img_fusion_rate = img_fusion*rate;

end