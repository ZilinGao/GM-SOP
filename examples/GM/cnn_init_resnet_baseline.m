function net = cnn_init_resnet_baseline(varargin)
%CNN_INIT_RESNET_BASELINE  Initialize the ResNet-50 model for ImageNet classification

opts.order = [];
opts.sectionLen = [];
opts.averageImage = zeros(3,1) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.modelType = [];
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

if ~(strcmp(opts.modelType,'ResNet18-GAP') ||  ...
        strcmp(opts.modelType, 'ResNet18-SR-SOP'))
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

%select pooling manner(SR-SOP or GAP) at the end of CNN
if opts.order == 2
    name1 =  'cov_pool';
    net.addLayer(name1, dagnn.OBJ_ConvNet_COV_Pool(), lastAdded.var,   name1) ;
    lastAdded.var = name1;
    
    name2 =  'cov_trace_norm';
    name_tr =  [name2 '_tr'];
    net.addLayer(name2 , dagnn.OBJ_ConvNet_Cov_TraceNorm(),  ...
        lastAdded.var,   {name2, name_tr}) ;
    lastAdded.var = name2;
    
    name3 = 'Cov_Sqrtm';
    net.addLayer(name3 , dagnn.OBJ_ConvNet_Cov_Sqrtm( 'coef', 1, 'iterNum', 3), ...
        lastAdded.var,   {name3, [name3 '_Y'], [name3, '_Z']}) ;
    lastAdded.var = name3;
    lastAdded.depth = lastAdded.depth * (lastAdded.depth + 1) / 2;
    
    name4 = 'Cov_ScaleTr';
    net.addLayer(name4 , dagnn.OBJ_ConvNet_COV_ScaleTr(),...
        {lastAdded.var, name_tr},  name4) ;
    lastAdded.var = name4;
else
    name = 'GAP';
    net.addLayer(name , ...
        dagnn.Pooling('poolSize', [8 8], 'method', 'avg'), ...
        lastAdded.var,name) ;
    lastAdded.var = name ;
end

% -------------------------------------------------------------------------
% Inference Layers
% -------------------------------------------------------------------------

net.addLayer('prediction' , ...
    dagnn.Conv('size', [1 1 lastAdded.depth  1000]), ...
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
    case 'ResNet18-GAP'
        lr =  0.75*[0.1 * ones(1,50), 0.01*ones(1,15), 0.001*ones(1,10)] ;
    case 'ResNet18-SR-SOP'
        lr =  1 * [0.1 * ones(1,40), 0.01*ones(1,10), 0.001*ones(1,5), 1e-4 * ones(1,5)] ;
end
folder = strcat(opts.modelType , '-LR_',num2str(lr(1)));

net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.batchSize = 128 * 2;
net.meta.trainOpts.numSubBatches = 1 ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.opt = folder ;
% Init parameters randomly
net.initParams() ;

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
