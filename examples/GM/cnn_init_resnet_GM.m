function net = cnn_init_resnet_GM(varargin)
%CNN_IMAGENET_INIT_RESNET  Initialize the ResNet-50 model for Gated Mixture
%Module (GM-GAP/GM-SOP)
%It is modified version based on cnn_init_resnet.m
%modified by Zilin Gao

opts.order = [];
opts.sectionLen = [];
opts.averageImage = zeros(3,1) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.topk = [];
opts.CM_num = [];
opts.loss_w = [];
opts.dropout = true;
opts.modelType = [];
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

if ~(strcmp(opts.modelType,'ResNet18-GM-GAP') |  ...
        strcmp(opts.modelType, 'ResNet18-GM-SOP'))
    error('illegal model type input')
end
lastAdded.var = 'input' ;
lastAdded.depth = 3 ;

    function Conv(name, ksize, depth, varargin)
        % Helper function to add a Convolutional + BatchNorm + ReLU
        % sequence to the network.
        args.relu = true ;
        args.downsample = false ;
        args.bias = false ;
        args = vl_argparse(args, varargin) ;
        if args.downsample, stride = 2 ; else stride = 1 ; end
        if args.bias, pars = {[name '_f'], [name '_b']} ; else pars = {[name '_f']} ; end
        net.addLayer([name  '_conv'], ...
            dagnn.Conv('size', [ksize ksize lastAdded.depth depth], ...
            'stride', stride, ....
            'pad', (ksize - 1) / 2, ...
            'hasBias', args.bias, ...
            'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
            lastAdded.var, ...
            [name '_conv'], ...
            pars) ;
        net.addLayer([name '_bn'], ...
            dagnn.BatchNorm('numChannels', depth), ...
            [name '_conv'], ...
            [name '_bn'], ...
            {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
        lastAdded.depth = depth ;
        lastAdded.var = [name '_bn'] ;
        if args.relu
            net.addLayer([name '_relu'] , ...
                dagnn.ReLU(), ...
                lastAdded.var, ...
                [name '_relu']) ;
            lastAdded.var = [name '_relu'] ;
        end
    end

    function Conv_nobn(name, ksize, depth, varargin)
        % Helper function to add a Convolutional +  ReLU
        % sequence to the network.
        args.relu = true ;
        args.downsample = false ;
        args.bias = false ;
        args = vl_argparse(args, varargin) ;
        if args.downsample, stride = 2 ; else stride = 1 ; end
        if args.bias, pars = {[name '_f'], [name '_b']} ; else pars = {[name '_f']} ; end
        net.addLayer([name  '_conv'], ...
            dagnn.Conv('size', [ksize ksize lastAdded.depth depth], ...
            'stride', stride, ....
            'pad', (ksize - 1) / 2, ...
            'hasBias', args.bias, ...
            'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
            lastAdded.var, ...
            [name '_conv'], ...
            pars) ;
        lastAdded.var = [name '_conv'] ;
        if args.relu
            net.addLayer([name '_relu'] , ...
                dagnn.ReLU(), ...
                lastAdded.var, ...
                [name '_relu']) ;
            lastAdded.var = [name '_relu'] ;
        end
    end


% -------------------------------------------------------------------------
% Add input section
% -------------------------------------------------------------------------

Conv('conv1', 3, 16, ...
    'relu', true, ...
    'bias', true, ...
    'downsample', false) ;

% -------------------------------------------------------------------------
% Add intermediate sections
% -------------------------------------------------------------------------

sectionLen = opts.sectionLen ;

for s = 2:5
    % -----------------------------------------------------------------------
    % Add intermediate segments for each section
    for l = 1:sectionLen
        depth = 2^(s+2) ;
        sectionInput = lastAdded ;
        name = sprintf('conv%d_%d', s, l)  ;
        
        % Optional adapter layer
        if l == 1 &&  s>2
            Conv([name '_adapt_conv'], 1, 2^(s+2), ...
                'downsample', s <5 | (s == 5 & opts.order == 1), 'relu', false) ;
        end
        sumInput = lastAdded ;
        
        % AB: 3x3, 3x3
        lastAdded = sectionInput ;
        Conv([name 'a'], 3, 2^(s+2), ...
            'downsample', (s == 3 | s==4 |(s == 5 & opts.order == 1)) & l == 1) ;
        %if pooling is 2nd-order, downsampling is not employed in stage 5
                
        Conv([name 'b'], 3, 2^(s+2) ,  'relu', false) ;
        
        % Sum layer
        net.addLayer([name '_sum'] , ...
            dagnn.Sum(), ...
            {sumInput.var, lastAdded.var}, ...
            [name '_sum']) ;
        net.addLayer([name '_relu'] , ...
            dagnn.ReLU(), ...
            [name '_sum'], ...
            name) ;
        lastAdded.var = name ;
    end
end

% -------------------------------------------------------------------------
% Gating Module
% -------------------------------------------------------------------------

last_var =  lastAdded.var;

%convolution layer dim in gating module
gating_dim1 = 256;
gating_dim2 = 128;

%dim of output in gating module
if opts.order == 2
    gating_dim_out = gating_dim2 * (gating_dim2 + 1)/2;
elseif opts.order == 1
    gating_dim_out = gating_dim2;
else
    error('illegal order input, support 1 or 2')
end

name = 'gating';
Conv([name '1'], 1, gating_dim1, 'relu', true) ;
Conv([name '2'], 1, gating_dim2, 'relu', true) ;

%select pooling manner(SR-SOP or GAP) in gating module
if opts.order == 2
    name_ = [name '_'];
    name1 = [name_ 'cov_pool'];
    net.addLayer(name1, dagnn.OBJ_ConvNet_COV_Pool(), lastAdded.var,   name1) ;
    lastAdded.var = name1;
    
    name2 = [name_ 'cov_trace_norm'];
    name_tr =  [name2 '_tr'];
    net.addLayer(name2 , dagnn.OBJ_ConvNet_Cov_TraceNorm(),  ...
        lastAdded.var,   {name2, name_tr}) ;
    lastAdded.var = name2;
    
    name3 = [name_ 'Cov_Sqrtm'];
    net.addLayer(name3 , dagnn.OBJ_ConvNet_Cov_Sqrtm( 'coef', 1, 'iterNum', 3), ...
        lastAdded.var,   {name3, [name3 '_Y'], [name3, '_Z']}) ;
    lastAdded.var = name3;
    lastAdded.depth = lastAdded.depth * (lastAdded.depth + 1) / 2;
    
    name4 = [name_ 'Cov_ScaleTr'];
    net.addLayer(name4 , dagnn.OBJ_ConvNet_COV_ScaleTr(),...
        {lastAdded.var, name_tr},  name4) ;
    lastAdded.var = name4;
else
    name = [name '_GAP'];
    net.addLayer(name , ...
        dagnn.Pooling('poolSize', [8 8], 'method', 'avg'), ...
        lastAdded.var,name) ;
    lastAdded.var = name ;
end

Hx_par = {'gating_w','gating_noise_w'} ;

net.addLayer('H_x' ,dagnn.H_x( ...
    'size' , [ gating_dim_out opts.CM_num] , ...
    'CM_num' , opts.CM_num , 'topk',opts.topk , ...
    'f_size'  , [1 1 ] ,'dim_in' , gating_dim_out),...
    lastAdded.var,'H_x',Hx_par );

net.addLayer('gating' , dagnn.gating( ...
    'size', [ gating_dim_out opts.CM_num] , ...
    'CM_num', opts.CM_num ,...
    'topk', opts.topk),...
    'H_x','gating');

loss_input = 'gating';

net.addLayer('Balance_loss',dagnn.Balance_loss(...
    'loss_weight',opts.loss_w ,'CM_num',...
    opts.CM_num),...
    loss_input,'Balance_loss');
net.vars(end).precious = true ;%save derivaition for balance loss backpropagation



% -------------------------------------------------------------------------
% Component Module(CM)
% -------------------------------------------------------------------------

switch opts.modelType
    case 'ResNet18-GM-GAP'
        CM_dim = 512;
    case 'ResNet18-GM-SOP'
        CM_dim = [256 128];
end

lastAdded.depth = 2^(s+2);

for b = 1:opts.CM_num
    lastAdded.var =  last_var;
    CM_name =  ['CM_' num2str(b) '_'];
    
    %ablation bn layer in each CM  
    for cc = 1: numel(CM_dim)
        Conv_nobn([CM_name ('a' + cc -1)] , 1, CM_dim(cc)) ;
        lastAdded.depth = CM_dim(cc);
    end
    
    if opts.order == 2
        name = [CM_name 'cov_pool'];
        net.addLayer(name , dagnn.OBJ_ConvNet_COV_Pool(), ...
            lastAdded.var,   name) ;
        lastAdded.var = name;
        
        name = [CM_name 'cov_trace_norm'];
        name_tr =  [name '_tr'];
        net.addLayer(name , dagnn.OBJ_ConvNet_Cov_TraceNorm(),  ...
            lastAdded.var,   {name, name_tr}) ;
        lastAdded.var = name;
        
        name = [CM_name 'Cov_Sqrtm'];
        net.addLayer(name , dagnn.OBJ_ConvNet_Cov_Sqrtm( 'coef', 1, 'iterNum', 3),  ...
            lastAdded.var,   {name, [name '_Y'], [name, '_Z']}) ;
        lastAdded.var = name;
        lastAdded.depth = lastAdded.depth * (lastAdded.depth + 1) / 2;
        
        name = [CM_name 'Cov_ScaleTr'];
        net.addLayer(name , dagnn.OBJ_ConvNet_COV_ScaleTr(),  ...
            {lastAdded.var, name_tr},  name) ;
    else
        net.addLayer([CM_name 'GAP'] , ...
            dagnn.Pooling('poolSize', [8 8], 'method', 'avg'), ...
            lastAdded.var,[CM_name 'GAP']) ;
    end
    
    lastAdded.depth = 2^(s+2) ;
end

% -------------------------------------------------------------------------
% Fusion by weighted sum
% -------------------------------------------------------------------------

in_{1} = 'gating'; %inputs of fusion layer
if opts.order == 2
    str_pool = '_Cov_ScaleTr';
else
    str_pool = '_GAP';
end

for ee = 1:opts.CM_num%----------------++++++++++++++++++++++++++++
    in_{ee + 1} = ['CM_', num2str(ee) , str_pool];
end


if opts.order == 2
    CM_out_dim  = CM_dim(end) * ( CM_dim(end) + 1) / 2;
else
    CM_out_dim = CM_dim(end);
end

net.addLayer( 'Fusion_out' , ...
    dagnn.CM_out('CM_num' ,opts.CM_num ,...
    'hdim' , CM_out_dim  ,...
    'topk' ,opts.topk),...
    in_ , 'Fusion_out' ) ;
lastAdded.var = 'Fusion_out' ;
lastAdded.depth = CM_out_dim;

folder = strcat(opts.modelType, '-' , num2str(opts.topk));
if opts.CM_num ~=opts.topk
    folder = strcat(folder,'from', num2str(opts.CM_num));
end

%add a bn layer after fusion operation
name = 'fusion_bn';
net.addLayer(name, ...
    dagnn.BatchNorm('numChannels', lastAdded.depth), ...
    lastAdded.var, ...
    name, ...
    {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
lastAdded.var = name;

if opts.dropout %add a dropout layer after fusion in case of overfitting
    name = 'dropout';
    drop_rate = 0.2;
    net.addLayer(name,dagnn.DropOut('rate',drop_rate),...
        lastAdded.var,name);
    lastAdded.var = name;
end


% -------------------------------------------------------------------------
% Inference Layers
% -------------------------------------------------------------------------

net.addLayer('prediction' , ...
    dagnn.Conv('size', [1 1 CM_out_dim  1000]), ...
    lastAdded.var, ...
    'prediction', ...
    {'prediction_f', 'prediction_b'}) ;

net.addLayer('loss', ...
    dagnn.Loss('loss', 'softmaxlog') ,...
    {'prediction', 'label'}, ...
    'objective') ;

net.addLayer('top1error', ...
    dagnn.Loss('loss', 'classerror'), ...
    {'prediction', 'label'}, ...
    'top1error') ;

net.addLayer('top5error', ...
    dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
    {'prediction', 'label'}, ...
    'top5error') ;


% -------------------------------------------------------------------------
%                                                           Meta
% -------------------------------------------------------------------------

net.meta.normalization.imageSize = [64 64 3] ;%for imagenet64 dataset
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.averageImage = opts.averageImage ;
net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;

switch opts.modelType
    case 'ResNet18-GM-GAP'
        lr =  0.75*[0.1 * ones(1,50), 0.01*ones(1,15), 0.001*ones(1,10)] ;
    case 'ResNet18-GM-SOP'
        lr =  1 * [0.1 * ones(1,40), 0.01*ones(1,10), 0.001*ones(1,5), 1e-4 * ones(1,5)] ;
end
folder = strcat(folder , '-LR_',num2str(lr(1)) , '_w_' , num2str(opts.loss_w));

net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.batchSize = 128 * 2;
net.meta.trainOpts.numSubBatches = 1 ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.opt = folder ;
% Init parameters randomly
net.initParams() ;

%--------------executuionLayers--------------------
% adjust execution order for this multi-branch model
if ~isnan(net.getLayerIndex('gating'))
    net.ExecutionOrder_GM();
end
%-------------------------------------------------

% For uniformity with the other ImageNet networks, t
% the input data is *not* normalized to have unit standard deviation,
% whereas this is enforced by batch normalization deeper down.
% The ImageNet standard deviation (for each of R, G, and B) is about 60, so
% we adjust the weights and learing rate accordingly in the first layer.
%
% This simple change improves performance almost +1% top 1 error.
p = net.getParamIndex('conv1_f') ;
net.params(p).value = net.params(p).value / 60 ;%trick
net.params(p).learningRate = net.params(p).learningRate / 60^2 ;%trick

for l = 1:numel(net.layers)
    if isa(net.layers(l).block, 'dagnn.BatchNorm')
        k = net.getParamIndex(net.layers(l).params{3}) ;
        net.params(k).learningRate = 0.3 ;
        net.params(k).epsilon = 1e-5 ;
    end
end

end
