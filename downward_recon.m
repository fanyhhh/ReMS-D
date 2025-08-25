function [img_fusion] = downward_recon(img_warped_same_area_cut, Rate_warped_same_area_cut, mask_warped_same_area_cut, calc_mode)

Patch_size_GWA=150;
Kernel=ones(Patch_size_GWA,Patch_size_GWA);
for img_num=1:length(img_warped_same_area_cut)
    for i=1:3
        I_locolmean(:,:,i)=imfilter(img_warped_same_area_cut{img_num}(:,:,i), Kernel,'symmetric')/sum(sum(Kernel));
    end
    I_locolmean_warped_same_area_cut{img_num}=I_locolmean;
end

for img_num=1:length(I_locolmean_warped_same_area_cut)
    mask_warped_same_area_cut{img_num}(:,:,1) = mask_warped_same_area_cut{img_num}(:,:,1) ./ mask_warped_same_area_cut{img_num}(:,:,1).^0.1;
    mask_warped_same_area_cut{img_num}(:,:,3) = mask_warped_same_area_cut{img_num}(:,:,3) .* mask_warped_same_area_cut{img_num}(:,:,3).^0.1;
end

iter_num=0;

for iter_num=1:4 % iter num can change
    if iter_num==1
        WB_type='no cali';
        if strcmp(WB_type,'no cali')
            channel_g0=zeros(1,length(I_locolmean_warped_same_area_cut));
            channel_g_rate=ones(1,length(I_locolmean_warped_same_area_cut));
            channel_b0=zeros(1,length(I_locolmean_warped_same_area_cut));
            channel_b_rate=ones(1,length(I_locolmean_warped_same_area_cut));
        else
            for img_num=1:length(I_locolmean_warped_same_area_cut)
                switch WB_type
                    case 'cali I/M'
                        channel_r=I_locolmean_warped_same_area_cut{img_num}(:,:,1)./mask_warped_same_area_cut{img_num}(:,:,1);
                        channel_g=I_locolmean_warped_same_area_cut{img_num}(:,:,2)./mask_warped_same_area_cut{img_num}(:,:,1);
                        channel_b=I_locolmean_warped_same_area_cut{img_num}(:,:,3)./mask_warped_same_area_cut{img_num}(:,:,1);
                    case 'cali I_mean'
                        channel_r=I_locolmean_warped_same_area_cut{img_num}(:,:,1);
                        channel_g=I_locolmean_warped_same_area_cut{img_num}(:,:,2);
                        channel_b=I_locolmean_warped_same_area_cut{img_num}(:,:,3);
                end

                fitresult = polyfit(channel_r(:), channel_g(:), 1);
                channel_g0(img_num)=fitresult(2);
                channel_g_rate(img_num)=fitresult(1);

                fitresult = polyfit(channel_r(:), channel_b(:), 1);
                channel_b0(img_num)=fitresult(2);
                channel_b_rate(img_num)=fitresult(1);

                g0=(channel_g-channel_g0(img_num))/channel_g_rate(img_num);
                b0=(channel_b-channel_b0(img_num))/channel_b_rate(img_num);

                disp([mean(channel_r(:)) mean(channel_g(:)) mean(channel_b(:)); mean(channel_r(:)) mean(g0(:)) mean(b0(:))]);

            end
        end

        IM=[];
        for img_num=1:length(I_locolmean_warped_same_area_cut)
            switch WB_type
                case 'no cali'
                    IM(:,:,:,img_num)=I_locolmean_warped_same_area_cut{img_num}./mask_warped_same_area_cut{img_num};
                case 'cali I/M'
                    IM(:,:,1,img_num)=I_locolmean_warped_same_area_cut{img_num}(:,:,1)./mask_warped_same_area_cut{img_num}(:,:,1);
                    IM(:,:,2,img_num)=(I_locolmean_warped_same_area_cut{img_num}(:,:,2)./mask_warped_same_area_cut{img_num}(:,:,2) - channel_g0(img_num)) / channel_g_rate(img_num);
                    IM(:,:,3,img_num)=(I_locolmean_warped_same_area_cut{img_num}(:,:,3)./mask_warped_same_area_cut{img_num}(:,:,3) - channel_b0(img_num)) / channel_b_rate(img_num);
                case 'cali I_mean'
                    IM(:,:,1,img_num)=I_locolmean_warped_same_area_cut{img_num}(:,:,1)./mask_warped_same_area_cut{img_num}(:,:,1);
                    IM(:,:,2,img_num)=((I_locolmean_warped_same_area_cut{img_num}(:,:,2) - channel_g0(img_num)) / channel_g_rate(img_num))./mask_warped_same_area_cut{img_num}(:,:,2);
                    IM(:,:,3,img_num)=((I_locolmean_warped_same_area_cut{img_num}(:,:,3) - channel_b0(img_num)) / channel_b_rate(img_num))./mask_warped_same_area_cut{img_num}(:,:,3);
            end
        end
        IM = permute(IM, [4, 3, 1, 2]);

        old_size = size(IM);
        new_size = [old_size(1:2), prod(old_size(3:end))];
        IM = reshape(IM, new_size);
        A0_min=max(IM,[],[1,3]);

        r=IM(:,1,:);
        g=IM(:,2,:);
        b=IM(:,3,:);

        r_sort=sort(r(:),'descend');
        g_sort=sort(g(:),'descend');
        b_sort=sort(b(:),'descend');

        A0_min_sort=[r_sort(round(length(r_sort)*0.001)),  g_sort(round(length(g_sort)*0.001)), b_sort(round(length(b_sort)*0.001))];

        solver_type='python';
        [A0] = A0_calc(I_locolmean_warped_same_area_cut, mask_warped_same_area_cut,A0_min_sort,channel_g0,channel_g_rate,channel_b0,channel_b_rate,WB_type,solver_type);

        solver_type='python';
        python_mode='separate';

        [row_result] = J_calc(I_locolmean_warped_same_area_cut, mask_warped_same_area_cut, Rate_warped_same_area_cut, A0,channel_g0,channel_g_rate,channel_b0,channel_b_rate,WB_type,0,solver_type,python_mode,calc_mode);

        for img_num=1:length(I_locolmean_warped_same_area_cut)
            switch WB_type
                case 'cali I/M'
                    I_locolmean_warped_same_area_cut{img_num}(:,:,1)=I_locolmean_warped_same_area_cut{img_num}(:,:,1);
                    I_locolmean_warped_same_area_cut{img_num}(:,:,2)=((I_locolmean_warped_same_area_cut{img_num}(:,:,2)./mask_warped_same_area_cut{img_num}(:,:,2) - channel_g0(img_num)) / channel_g_rate(img_num)).*mask_warped_same_area_cut{img_num}(:,:,2);
                    I_locolmean_warped_same_area_cut{img_num}(:,:,3)=((I_locolmean_warped_same_area_cut{img_num}(:,:,3)./mask_warped_same_area_cut{img_num}(:,:,3) - channel_b0(img_num)) / channel_b_rate(img_num)).*mask_warped_same_area_cut{img_num}(:,:,3);
                    img_warped_same_area_cut{img_num}(:,:,1)=img_warped_same_area_cut{img_num}(:,:,1);
                    img_warped_same_area_cut{img_num}(:,:,2)=((img_warped_same_area_cut{img_num}(:,:,2)./mask_warped_same_area_cut{img_num}(:,:,2) - channel_g0(img_num)) / channel_g_rate(img_num)).*mask_warped_same_area_cut{img_num}(:,:,2);
                    img_warped_same_area_cut{img_num}(:,:,3)=((img_warped_same_area_cut{img_num}(:,:,3)./mask_warped_same_area_cut{img_num}(:,:,3) - channel_b0(img_num)) / channel_b_rate(img_num)).*mask_warped_same_area_cut{img_num}(:,:,3);
                case 'cali I_mean'
                    I_locolmean_warped_same_area_cut{img_num}(:,:,1)=I_locolmean_warped_same_area_cut{img_num}(:,:,1);
                    I_locolmean_warped_same_area_cut{img_num}(:,:,2)=(I_locolmean_warped_same_area_cut{img_num}(:,:,2) - channel_g0(img_num)) / channel_g_rate(img_num);
                    I_locolmean_warped_same_area_cut{img_num}(:,:,3)=(I_locolmean_warped_same_area_cut{img_num}(:,:,3) - channel_b0(img_num)) / channel_b_rate(img_num);
                    img_warped_same_area_cut{img_num}(:,:,1)=img_warped_same_area_cut{img_num}(:,:,1);
                    img_warped_same_area_cut{img_num}(:,:,2)=(img_warped_same_area_cut{img_num}(:,:,2) - channel_g0(img_num)) / channel_g_rate(img_num);
                    img_warped_same_area_cut{img_num}(:,:,3)=(img_warped_same_area_cut{img_num}(:,:,3) - channel_b0(img_num)) / channel_b_rate(img_num);
            end
        end
        [row_iter,J0_all] = warped_data_dehaze(A0,row_result,mask_warped_same_area_cut,I_locolmean_warped_same_area_cut,Rate_warped_same_area_cut,img_warped_same_area_cut);
    else
        [row_iter,J0_all] = warped_data_dehaze(A0,row_iter,mask_warped_same_area_cut,I_locolmean_warped_same_area_cut,Rate_warped_same_area_cut,img_warped_same_area_cut);
    end


[img_fusion] = downward_fusion(I_locolmean_warped_same_area_cut,Rate_warped_same_area_cut,img_warped_same_area_cut,mask_warped_same_area_cut,J0_all,A0);
img_fusion = img_fusion .* mean(img_fusion(:)) ./ mean(img_fusion,[1,2]);
light = 0.25;
img_fusion = img_fusion * light / mean(img_fusion(:));

end
