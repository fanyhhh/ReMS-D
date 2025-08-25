clear all
close all

%% data load
filefold = ['..\ReMS-D\forward facing\30\'];
filename = [filefold, 'img.mat'];
load(filename);
filename = [filefold, 'direction_para.mat'];
load(filename);

%% processing
% calc_mode='cpu'; % high quality
calc_mode='gpu'; % fast

% [img_fusion] = downward_recon(img, R, M, calc_mode);
[img_fusion] = forward_recon(img, R, M, calc_mode);

figure;
imshow(img_fusion);