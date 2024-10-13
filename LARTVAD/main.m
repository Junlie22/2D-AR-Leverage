%% Demo for LARTVAD
% Paper: Hyperspectral anomaly detection with tensor average rank and piecewise smoothness constraints, IEEE TNNLS
% Authors: Siyu Sun, Jun Liu*, Xun Chen, Wei Li, and Hongbin Li
% Time: 2022-11-04
% All Rights Reserved,
% Email: junliu@ustc.edu.cn

clear all;
% range of parameters
% Beta    = [0.01,0.1,1,2,3,10];
% Lambda  = [0.01,0.1,1,2,3,10];
% K       = 60:20:200;
addpath(genpath('Function'))
addpath(genpath('tensor_toolbox_2.5'))

%FileName        = {'../dataset/Airport', '../dataset/San_Diego'};
FileName        = {'p011_neutral','p011_smile','p015_neutral'};

LARTVAD_K       = 80; % the number of superpixels
LARTVAD_beta    = 10;
LARTVAD_lambda  = 2;

LARTVAD_AS = struct;

for file = 1:length(FileName)
    filename            = FileName{file};
    DATA                = load(filename);
    data                = DATA.data;
    %map                 = DATA.map;
    tic
    data                = (data-min(data(:))) / (max(data(:))-min(data(:)));    
    LARTVAD_Dic         = DictConstruct(data,LARTVAD_K);  % dictionary
    S                   = LARTVAD(data,LARTVAD_Dic,LARTVAD_beta,LARTVAD_lambda);
    R_LARTVAD           = sqrt(sum(S.^2,3));
    R_LARTVAD           = (R_LARTVAD-min(R_LARTVAD(:)))./(max(R_LARTVAD(:))-min(R_LARTVAD(:)));
    %[PF_LARTVAD,PD_LARTVAD,Tau_LARTVAD] = perfcurve(map(:),R_LARTVAD(:),'1') ;
    %AUC_LARTVAD         = -sum((PF_LARTVAD(1:end-1)-PF_LARTVAD(2:end)).*(PD_LARTVAD(2:end) + PD_LARTVAD(1:end-1))/2);
    %AUC_LARTVAD_PDtau   = sum((Tau_LARTVAD(1:end-1)-Tau_LARTVAD(2:end)).*(PD_LARTVAD(2:end)+PD_LARTVAD(1:end-1))/2);
    %AUC_LARTVAD_PFtau   = sum((Tau_LARTVAD(1:end-1)-Tau_LARTVAD(2:end)).*(PF_LARTVAD(2:end)+PF_LARTVAD(1:end-1))/2);
    %AUC_LARTVAD_OD      = AUC_LARTVAD + AUC_LARTVAD_PDtau - AUC_LARTVAD_PFtau;
    %AUC_LARTVAD_SNR     = AUC_LARTVAD_PDtau / AUC_LARTVAD_PFtau;
    %show_LARTVAD        = R_LARTVAD;
    t_LARTVAD           = toc;   

    LARTVAD_AS(file).name = filename;
    LARTVAD_AS(file).as = R_LARTVAD;
    LARTVAD_AS(file).time = t_LARTVAD;
end

save LARTVAD__hyper_skin.mat LARTVAD_AS;

