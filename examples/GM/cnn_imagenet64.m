function [net, info] = cnn_imagenet64(varargin)
%CNN_IMAGENET64  The main funcion of work Global Gated Mixture of Second-order Pooling for
% Improving Deep Convolutional Neural Networks.
%
% Modified on cnn_imagenet.m
% Modified by Zilin Gao

run(fullfile(fileparts(mfilename('fullpath')), ...
    '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.modelType = 'ResNet18-GM-GAP';
% 'ResNet18-GM-GAP' 'ResNet18-GM-GOP' 'WRN-36-2-GM-GAP' 'WRN-36-2-GM-SOP'
% 'ResNet18-GAP'  'ResNet18-SR-SOP'
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
opts.expDir =  fullfile(vl_rootnn, 'data');
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;

opts.imdbPath = 'imdb.mat';%fake imdb for debug(with quick load process)

opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
opts.train.gpus = [2];
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;


switch opts.modelType
    case 'ResNet18-GM-GAP' 
        opts.topk = 8;
        opts.CM_num = 16;
        opts.loss_w = 100;
        opts.order = 1;%1:GAP  2:SOP
        opts.sectionLen = 2;% (layer = 8 * sectionLen + 2)
        opts.dropout = true;
        
    case 'ResNet18-GM-SOP' 
        opts.topk = 8;
        opts.CM_num = 16;
        opts.loss_w = 100;
        opts.order = 2;
        opts.sectionLen = 2;
        opts.dropout = true;
        
    case 'ResNet18-GAP'
        opts.order = 1;
        opts.sectionLen = 2;
        
    case 'ResNet18-SR-SOP'
        opts.order = 2;
        opts.sectionLen = 2;
        
    case 'WRN-36-2-GM-GAP' 
        opts.topk = 8;
        opts.CM_num = 16;
        opts.loss_w = 100;
        opts.order = 1;
        opts.dropout = true;
        opts.sectionLen = 4;
        
    case 'WRN-36-2-GM-SOP'
        opts.topk = 8;
        opts.CM_num = 16;
        opts.loss_w = 100;
        opts.order = 2;
        opts.dropout = true;
        opts.sectionLen = 4;
        
    otherwise
        error('illegal net type input')
end


% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
    disp(['loading imdb.mat into RAM...'])
    imdb = load(opts.imdbPath) ;
    disp(['loading over'])
    if isfield(imdb, 'imdb')
        imdb = imdb.imdb;
    end
else
    error('imdb not exist in assigned path')
end


% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

if ~isempty(strfind(opts.modelType,'GM'))%GATED-MIXTURE MODEL
    if ~isempty(strfind(opts.modelType,'ResNet'))
        net = cnn_init_resnet_GM('averageImage',...
            imdb.meta.rgbMean , ...
            'order' , opts.order , 'sectionLen' , opts.sectionLen ,...
            'topk',opts.topk, 'CM_num',opts.CM_num, ...
            'loss_w',opts.loss_w,...
            'dropout',opts.dropout,...
            'modelType',opts.modelType) ;
        
    elseif ~isempty(strfind(opts.modelType,'WRN-36-2'))
         net = cnn_init_WRN_GM('averageImage',...
            imdb.meta.rgbMean , ...
            'order' , opts.order , 'sectionLen' , opts.sectionLen ,...
            'topk',opts.topk, 'CM_num',opts.CM_num, ...
            'loss_w',opts.loss_w,...
            'dropout',opts.dropout,...
            'modelType',opts.modelType) ;       
    end
    
elseif ~isempty(strfind(opts.modelType,'GAP')) || ...
    ~isempty(strfind(opts.modelType,'SR-SOP')) %baseline
    net = cnn_init_resnet_baseline('averageImage',...%GAP or SR-SOP
            imdb.meta.rgbMean , ...
            'order' , opts.order , 'sectionLen' , opts.sectionLen,...
            'modelType',opts.modelType);
end
opts.networkType = 'dagnn' ;
folder_name = net.meta.opt;
opts.expDir = fullfile(opts.expDir ,folder_name );


% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
    case 'simplenn', trainFn = @cnn_train ;
    case 'dagnn', trainFn = @cnn_train_GM_dag ;
end

[net, info] = trainFn(net, imdb, getBatch_Img64(opts), ...
    'expDir', opts.expDir, ...
    net.meta.trainOpts, ...
    opts.train , 'order' , opts.order) ;


% -------------------------------------------------------------------------
function fn = getBatch_Img64(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
    case 'simplenn'
        fn = @(x,y) getSimpleNNBatch(x,y) ;
    case 'dagnn'
        bopts = struct('numGpus', numel(opts.train.gpus)) ;
        
        fn = @(x,y,m_,mode) ...
            getDagNNBatch(bopts,x,y,m_,mode) ;
end


% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch,varargin)
% -------------------------------------------------------------------------
jitter.mode = 'train';
jitter.flip = true ;
jitter = vl_argparse(jitter, varargin) ;

images_org = single(imdb.images.data(:,:,:,batch) );
labels = single(imdb.images.labels(1,batch) );

%subtract mean
images_sub = single(bsxfun(@minus , images_org , ...
    reshape(imdb.meta.rgbMean , [1,1,3,1])) );

if strcmp( jitter.mode, 'train')
    if jitter.flip
        %random flip for each img
        kk = rand(1,size(images_org , 4));
        flip_num = find( ( kk > 0.5) == 1);
        images = images_sub;
        images(:,:,:,flip_num) = fliplr(images_sub(:,:,:,flip_num));
        
        % %random flip for each batch
        % if rand > 0.5, images=fliplr(images) ; end
    end
else
    images = images_sub;
end

if opts.numGpus > 0
    images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;



