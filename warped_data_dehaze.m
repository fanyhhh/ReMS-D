function [row_iter,J0_all] = warped_data_dehaze(a_result,row_result,mask_warped_same_area_cut,I_locolmean_warped_same_area_cut,Rate_warped_same_area_cut,img_warped_same_area_cut)

for img_num=1:length(img_warped_same_area_cut)
    t_iter{img_num}=ones(size(img_warped_same_area_cut{img_num}))*0.5;
end

for img_num=1:length(img_warped_same_area_cut)
    for k=1:1
        for i=1:3
            A0(:,:,i)=a_result(i);
        end
        A=mask_warped_same_area_cut{img_num}.*A0;
                
        t = ones(size(A))*0.5;
        f = @(x) I_locolmean_warped_same_area_cut{img_num} - (Rate_warped_same_area_cut{img_num}.*A0.*row_result{img_num}.*x + A.*(1-x)); % 增加反照
        tol = 1e-6;
        maxiter = 10;

        step = 1e-6;
        t = newton_t(f, t, tol, step, maxiter);
        t=min(max(t,0),1);
        
        t_iter{img_num}=t;       
        
        row = ones(size(A))*0.5;
        f = @(x) img_warped_same_area_cut{img_num} - (Rate_warped_same_area_cut{img_num}.*A0.*x.*t_iter{img_num} + A.*(1-t_iter{img_num})); % 增加反照
        tol = 1e-6;
        maxiter = 10;
        
        step = 1e-6;
        row = newton_row(f, row, tol, step, maxiter);
        row_iter{img_num}=row;
                
        J=row.*A0;
        
        mr=A0(:,:,1);
        mg=A0(:,:,2);
        mb=A0(:,:,3);
        J0(:,:,1)=J(:,:,1)*mean([mr,mg,mb])/mr;
        J0(:,:,2)=J(:,:,2)*mean([mr,mg,mb])/mg;
        J0(:,:,3)=J(:,:,3)*mean([mr,mg,mb])/mb;
 
        adj_percent = [0.005, 0.995];
        contrast_limit=stretchlim(J0,adj_percent);
        J0=min(max(J0,min(contrast_limit(1,:))),max(contrast_limit(2,:)));
        
        contrast_limit=stretchlim(J,adj_percent);
        J=min(max(J,min(contrast_limit(1,:))),max(contrast_limit(2,:)));
    end
    
    J0_all{img_num}=J0;

end
end