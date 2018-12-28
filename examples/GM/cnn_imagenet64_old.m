function [net, info] = cnn_imagenet(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%  This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%  VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.


run(fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;
dataset = 'ImageNet64';
opts.dataDir = fullfile('s:', 'dataset',dataset) ;
opts.modelType = 'resnet-3x3' ;
% 'CONV-GATE';% 'GAP-IN-WXGATING';% 'resnet-3x3' ;%'resnet-50' 'resnet-74'
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.order =1;%1:avgpooling 2:MPN-COV 9:architecture test
opt.sectionLen = 2 ;%2:resnet-18 600hz ; 4:resnet-34 300hz (for 1st order)
% opts.o2p_method  ='MPN-COV-MATLAB';%'MPN-COV-MATLAB';% 'SVD-PN';%iSQRT or MPN-COV o[r SVD-PN
opts.o2p_method  = 'SVD-PN' ;
% opts.comb = 'topk+l2+sf' ; 
opts.wx_method ='noise';%'noise';%'wx';
opts.gating = 'topk-sf';% 'topk-sf';
opts.comb = 'topk-sf';%'topk-sf' ;% ; % 'topk-sf';%'no_wb' ; 
opts.loss_comb ='none';%'no-loss';%'im-l2-sf‘ 'sf';%'no_wb';
opts.extra_loss ='importance';%'importance'or [];
opts.w_init = 'ones';%'randn','ones';
opts.loss_w = 50;
opts.topk = 4;
opts.branch_num = 4;
opts.noise_init  = 'unifrnd';

% opts.comb = 'softmax' ;


opts.coloraug = false;

if any(strncmp(opts.modelType ,{'resnet'}, 6))
    opts.networkType = 'dagnn' ;
end

opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
% opts.expDir = fullfile(vl_rootnn, 'data', ['imagenet12-' sfx]) ;
opts.expDir =  fullfile(vl_rootnn, 'data');% , strcat(dataset,'-' ,opts.modelType));
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;

opts.imdbPath = '/media/hh/SSD256/ImageNet64/imdb.mat';
opts.imdbPath = '/media/hh/SSD256/ImageNet64/imdb_b1_addval.mat'; 

% opts.imdbPath = 'imdb.mat';

opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
opts.train.gpus = [1];
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
    disp(['loading imdb.mat into RAM...'])
    imdb = load(opts.imdbPath) ;
    disp(['loading over'])
    %   imdb.imageDir = fullfile(opts.dataDir, 'images');
else
    imdb = getImageNet64Imdb(opts.dataDir,false) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% Compute image statistics (mean, RGB covariances, etc.)
% imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
% if exist(imageStatsPath)
%   load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
% else
%     error('ImageState missing')
% %   train = find(imdb.images.set == 1) ;
% %   images = fullfile(imdb.imageDir, imdb.images.name(train(1:100:end))) ;
% %
% %   [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
% %                                                     'imageSize', [256 256], ...
% %                                                     'numThreads', opts.numFetchThreads, ...
% %                                                     'gpus', opts.train.gpus) ;
% %   save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
% end

if opts.coloraug
    covariance_path = fullfile('D:','Database','ImageNet64', 'rgbCovariance.mat');
    if ~exist(covariance_path)
         rgbCovariance = getImageNet64Covariance(imdb );
         save(covariance_path , 'rgbCovariance' );
    else
          rgbCovariance = load(covariance_path);
          if isfield(rgbCovariance , 'rgbCovariance')
              rgbCovariance = rgbCovariance.rgbCovariance;
          end
    end
    % prepare for color jittering
%     rgbCovariance = imdb.meta.rgbCovariance;
    [v,d] = eig(rgbCovariance) ;
    rgbDeviation =  v*sqrt(d) ; %brightness_jitter:b
    clear v d ;
else
    rgbDeviation = zeros(3);
end

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

% %       net = cnn_imagenet_init_resnet('averageImage', rgbMean, ...
% %                                      'colorDeviation', rgbDeviation, ...
% %                                      'classNames', imdb.classes.name, ...
% %                                      'classDescriptions', imdb.classes.description) ;
% %       opts.networkType = 'dagnn' ;

if isempty(opts.network)
    switch opts.modelType
        %   if strncmp(opts.modelType , {'resnet'} ,6)
        %     imdb.meta.rgbMean = rand(1,3)*255;%++++++++++++++++++++
        case 'resnet-3x3'
          net = cnn_MoE_init_resnet('averageImage',...
                    imdb.meta.rgbMean , 'colorDeviation', rgbDeviation , ...
                    'order' , opts.order , 'sectionLen' , opt.sectionLen ,...
                    'o2p_method' , opts.o2p_method , ...
                    'comb_method' , opts.comb,'topk',opts.topk,...
                    'branch_num',opts.branch_num,'loss',opts.extra_loss,...
                    'loss_w',opts.loss_w,...
                     'loss_comb_method' ,  opts.loss_comb,...
                     'extra_loss',opts.extra_loss,'wx_method',opts.wx_method,...
                     'gating',opts.gating,'w_init',opts.w_init,...
                     'noise_init',opts.noise_init ) ;
            opts.networkType = 'dagnn' ;                
        case 'GAP-IN-WXGATING'
           net = cnn_MoE_POOL_IN_WXGATE_init_resnet('averageImage',...
                    imdb.meta.rgbMean , 'colorDeviation', rgbDeviation , ...
                    'order' , opts.order , 'sectionLen' , opt.sectionLen ,...
                    'o2p_method' , opts.o2p_method , ...
                    'comb_method' , opts.comb,'topk',opts.topk,...
                    'branch_num',opts.branch_num,'loss',opts.extra_loss,...
                    'loss_w',opts.loss_w,...
                     'loss_comb_method' ,  opts.loss_comb,...
                     'extra_loss',opts.extra_loss,'wx_method',opts.wx_method,...
                     'gating',opts.gating) ;           
            opts.networkType = 'dagnn' ;
        case 'loss_only'
           net = cnn_MoE_POOL_IN_WXGATE_init_resnet_loss('averageImage',...
                    imdb.meta.rgbMean , 'colorDeviation', rgbDeviation , ...
                    'order' , opts.order , 'sectionLen' , opt.sectionLen ,...
                    'o2p_method' , opts.o2p_method , ...
                    'comb_method' , opts.comb,'topk',opts.topk,...
                    'branch_num',opts.branch_num,'loss',opts.extra_loss,...
                    'loss_w',opts.loss_w,...
                     'loss_comb_method' ,  opts.loss_comb,...
                     'extra_loss',opts.extra_loss,'wx_method',opts.wx_method,...
                     'gating',opts.gating) ;           
            opts.networkType = 'dagnn' ;            
        case 'CONV-GATE'
             net = cnn_MoE_CONV_GATE_init_resnet('averageImage',...
                    imdb.meta.rgbMean , 'colorDeviation', rgbDeviation , ...
                    'order' , opts.order , 'sectionLen' , opt.sectionLen ,...
                    'o2p_method' , opts.o2p_method , ...
                    'comb_method' , opts.comb,'topk',opts.topk,...
                    'branch_num',opts.branch_num,'loss',opts.extra_loss,...
                    'loss_w',opts.loss_w,...
                     'loss_comb_method' ,  opts.loss_comb,...
                     'extra_loss',opts.extra_loss,'wx_method',opts.wx_method,...
                     'gating',opts.gating) ;  
               opts.networkType = 'dagnn' ;                         
        case 'resnet-50'
            net = cnn_imagenet64_init_resnet50('averageImage',imdb.meta.rgbMean) ;
            opts.networkType = 'dagnn' ;
        otherwise
            net = cnn_imagenet_init('model', opts.modelType, ...
                'batchNormalization', opts.batchNormalization, ...
                'weightInitMethod', opts.weightInitMethod, ...
                'networkType', opts.networkType, ...
                'averageImage', rgbMean, ...
                'colorDeviation', rgbDeviation, ...
                'classNames', imdb.classes.name, ...
                'classDescriptions', imdb.classes.description) ;
    end
else
    net = opts.network ;
    opts.network = [] ;
end

% opts.expDir = fullfile(opts.expDir ,strcat('order' , num2str(opts.order)), strcat('LR' , num2str(net.meta.trainOpts.learningRate(1) )));
% % % folder_name = strcat('ResNet' , num2str(opt.sectionLen * 8 +2 ) , '_Order' ,...
% % %     num2str(opts.order) , '_LR' , num2str(net.meta.trainOpts.learningRate(1)));
%12.11

% if opts.order <2
% folder_name = strcat('ResNet' , num2str(opt.sectionLen * 8 +2 ) , '_Order' , ...
%     num2str(opts.order) , '_' ,num2str(opts.topk),'in',num2str(opts.branch_num),...
%     '_', opts.comb ,'_LR' , num2str(net.meta.trainOpts.learningRate(1)));
% 
% elseif opts.order == 2
%     folder_name = strcat('ResNet' , num2str(opt.sectionLen * 8 +2 ) , '_Order' ,...
%     num2str(opts.order) ,'_', opts.o2p_method ,'_LR' , num2str(net.meta.trainOpts.learningRate(1)));
% end
folder_name = net.meta.opt;
opts.expDir = fullfile(opts.expDir ,folder_name );


% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
    case 'simplenn', trainFn = @cnn_train ;
    case 'dagnn', trainFn = @cnn_train_img64_dag ;
end

[net, info] = trainFn(net, imdb, getBatch_Img64(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train , 'order' , opts.order) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

% net = cnn_imagenet_deploy(net) ;
% modelPath = fullfile(opts.expDir, 'net-deployed.mat')
%
% switch opts.networkType
%   case 'simplenn'
%     save(modelPath, '-struct', 'net') ;
%   case 'dagnn'
%     net_ = net.saveobj() ;
%     save(modelPath, '-struct', 'net_') ;
%     clear net_ ;
% end

% -------------------------------------------------------------------------
function fn = getBatch_Img64(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
    case 'simplenn'
        fn = @(x,y) getSimpleNNBatch(x,y) ;
    case 'dagnn'
        bopts = struct('numGpus', numel(opts.train.gpus)) ;
        
        fn = @(x,y,flag_c_n,flag_c_p,b_n,b_p,c_n,c_p,s_c,s_p) ...
            getDagNNBatch(bopts,x,y,flag_c_n,flag_c_p,...
            b_n,b_p,c_n,c_p,s_c,s_p) ;
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch,varargin)
% -------------------------------------------------------------------------

jitter.color = false;
jitter.flip = true ;
jitter.brightness = zeros(3,1);
jitter.saturation = 0.3;
jitter.contrast = 0.2;
jitter = vl_argparse(jitter, varargin) ;

images_org = single(imdb.images.data(:,:,:,batch) );
labels = single(imdb.images.labels(1,batch) );

if jitter.color
for pic_num = 1:numel(batch)
    img = images_org(:,:,:,pic_num);
% % %color jitter as in imagenet

    %brightness jitter
    w = randn(3,1) ; 
     brightness_rand = jitter.brightness * w;
%       w = [-0.3; 0.4; 0.05];
%      b_temp.brightness = [0.1 0.2 -0.07; 1.3 -0.8 0.5; 0.6 -1.5 -2.3] ;
%      brightness_rand = b_temp.brightness * w;
    brightness_shift = brightness_rand -  imdb.meta.rgbMean;%Bw - u
     dv_b = reshape(brightness_shift , 1,1,3);
    
    avg_img = mean( mean(img , 1 ) ,2);
    %contrast
    contrast_shift = unifrnd(1 - jitter.contrast , 1 + jitter.contrast);%rand???
%     contrast_shift = 0.7;
    dv_b_c =  (1 - contrast_shift) * avg_img + dv_b ;

    x_jitter = bsxfun(@plus , contrast_shift * img , dv_b_c);%64*64*3
    
     saturation_shift = unifrnd(1-jitter.saturation , 1 + jitter.saturation);
%      saturation_shift = 0.9;
     
    jitter_factor = saturation_shift * eye(3) + (1-saturation_shift) * ones(3,3) ./ 3;

    temp_1 = reshape(permute(x_jitter  , [3 1 2]), 3 , []);
    temp_2 = jitter_factor * temp_1 ; 
    images_sub(:,:,:,pic_num)= reshape(permute(temp_2 , [2 1]) , 448 , 448 , 3);
    
% 
end
else
    %subtract mean
    images_sub = single(bsxfun(@minus , images_org , reshape(imdb.meta.rgbMean , [1,1,3,1])) );
end

if jitter.flip
    %random flip for each pic
    kk = rand(1,size(images_org , 4));
    flip_num = find( ( kk > 0.5) == 1);
    images = images_sub;
    images(:,:,:,flip_num) = fliplr(images_sub(:,:,:,flip_num));
    
    % % % %random flip for each batch
    % % % if rand > 0.5, images=fliplr(images) ; end
end

if opts.numGpus > 0
    images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;


% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

if numel(meta.normalization.averageImage) == 3
    mu = double(meta.normalization.averageImage(:)) ;
else
    mu = imresize(single(meta.normalization.averageImage), ...
        meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
    'useGpu', useGpu, ...
    'numThreads', opts.numFetchThreads, ...
    'imageSize',  meta.normalization.imageSize(1:2), ...
    'cropSize', meta.normalization.cropSize, ...
    'subtractAverage', mu) ;

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
for f = fieldnames(meta.augmentation)'
    f = char(f) ;
    bopts.train.(f) = meta.augmentation.(f) ;
end

fn = @(x,y) getBatch(bopts,useGpu,lower(opts.networkType),x,y) ;


function rgbCovariance = getImageNet64Covariance(imdb  )
%rgb_covariance
pic_num = randperm(sum(imdb.images.set == 1) , 10000);%get 1e4 random sample from train set
sample  = imdb.images.data(:,:,:,pic_num);
sample = permute(sample , [3 1 2 4]);
rgb = double(reshape(sample , 3 ,[]));
rgb1 = mean(rgb , 2);
rgb2 = rgb * rgb' / size(rgb , 2);
rgbCovariance = rgb2 - rgb1 * rgb1';

% imdb.meta.rgbCovariance = rgbCovariance;

function imdb = getImageNet64Imdb(dataDir , contrastNormalization)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
%contrast normalization ? whiten data?
imdb_path = fullfile(dataDir , 'imdb.mat');

if ~exist(imdb_path)
    imdb = permute_pic(dataDir);
else
    imdb = load(imdb_path);
end

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng
%???
if contrastNormalization
    z = reshape(data,[],60000) ;
    z = bsxfun(@minus, z, mean(z,1)) ;
    n = std(z,0,1) ;
    z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
    data = reshape(z, 32, 32, 3, []) ;
end




% % if opts.whitenData
% %   z = reshape(data,[],60000) ;
% %   W = z(:,set == 1)*z(:,set == 1)'/60000 ;
% %   [V,D] = eig(W) ;
% %   % the scale is selected to approximately preserve the norm of W
% %   d2 = diag(D) ;
% %   en = sqrt(mean(d2)) ;
% %   z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
% %   data = reshape(z, 32, 32, 3, []) ;
% % end

% clNames = load(fullfile(unpackPath, 'batches.meta.mat'));


% imdb.images.data = data ;
% imdb.images.labels = single(cat(2, labels{:})) ;
% imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
% imdb.meta.classes = clNames.label_names;

function imdb = permute_pic(dataDir )
%dataDir = fullfile('s:' , 'dataset'  , 'ImageNet64');
files = [arrayfun(@(n) sprintf('b%d.mat', n), 1:10, 'UniformOutput', false) ...
    {'val.mat'}];
for i=1:11
    file_path = fullfile(dataDir , files{i});
    data_total = load(file_path);
    if i==1
        mean = data_total.mean;
    end
    data = data_total.data;
    data_rev = data';
    [pixel pic] = size(data_rev);
    pic_rev = reshape(data_rev , [sqrt(pixel / 3) sqrt(pixel / 3) 3 pic]);%[col row channel pic]
    DATA{i} = permute(pic_rev , [2 1 3 4]);%[row col channel pic]
    label{i} = data_total.labels;
end

delete data data_rev pic_rev
merge_temp = {};
LABEL = {};
val_num = numel(label{11});
MEAN = reshape(mean , 64,64,3);
for i=[1:11]
    DATA{i} = bsxfun(@minus, DATA{i}, MEAN);
    merge_temp = {cell2mat(cat(4 ,merge_temp ,  DATA{i}))};
    DATA{i} = [];
    LABEL = {cell2mat(cat(2 , LABEL ,label{i} ))};
    
    label{i} = [];
end

% remove mean in any case

data_sub_mean = bsxfun(@minus, merge_temp{1}, MEAN);

clear merge_temp

imdb.images.data = data_sub_mean;
imdb.images.labels = LABELS{1};
imdb.images.sets = [ones(total_num - val_num , 1) ; 2*ones(val_num , 1)];
% imdb.images.mean = mean;

save(fullfile(dataDir ,'imdb.mat' ),  'imdb');

% -------------------------------------------------------------------------
function inputs = getImageNet64_Batch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
    images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;