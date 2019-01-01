function net = cnn_imagenet64_init_resnet(varargin)
%CNN_IMAGENET_INIT_RESNET  Initialize the ResNet-50 model for ImageNet64 classification

opts.order = 1;
opts.sectionLen = 4;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.o2p_method = [];
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

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

% -------------------------------------------------------------------------
% Add input section
% -------------------------------------------------------------------------

Conv('conv1', 3, 16, ...
    'relu', true, ...
    'bias', true, ...
    'downsample', false) ;

% % % net.addLayer(...
% % %     'conv1_pool' , ...
% % %     dagnn.Pooling('poolSize', [3 3], ...
% % %     'stride', 1, ...
% % %     'pad', 1,  ...
% % %     'method', 'max'), ...
% % %     lastAdded.var, ...
% % %     'conv1') ;
% % % lastAdded.var = 'conv1' ;%12.7

% -------------------------------------------------------------------------
% Add intermediate sections
% -------------------------------------------------------------------------

sectionLen = opts.sectionLen ;  ;%9 : 74layers

for s = 2:5
    %     sectionLen = 4  ;%9 : 74layers
    %   switch s
    %     case 2, sectionLen = 3 ;
    %     case 3, sectionLen = 4 ; % 8 ;
    %     case 4, sectionLen = 6 ; % 23 ; % 36 ;
    %     case 5, sectionLen = 3 ;
    %   end
    
    % -----------------------------------------------------------------------
    % Add intermediate segments for each section
    for l = 1:sectionLen
        depth = 2^(s+2) ;%s 2:16
        sectionInput = lastAdded ;
        name = sprintf('conv%d_%d', s, l)  ;
        
        % Optional adapter layer
        if l == 1 & s>2
            % %    if l == 1
            Conv([name '_adapt_conv'], 1, 2^(s+2), ...
                'downsample', s <5 | (s == 5 & opts.order == 1), 'relu', false) ;
        end
        sumInput = lastAdded ;
        
        % ABC: 1x1, 3x3, 1x1; downsample if first segment in section from
        % section 2 onwards.
        lastAdded = sectionInput ;
        Conv([name 'a'], 3, 2^(s+2), ...
            'downsample', (s == 3 | s==4 |(s == 5 & opts.order == 1)) & l == 1) ;
        Conv([name 'b'], 3, 2^(s+2) ,  'relu', false) ;
        %     Conv([name 'c'], 1, 2^(s+6), 'relu', false) ;
        
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

switch opts.order
    
    case 1
        name =  'prediction_avg';
        % % %         net.addLayer('prediction_avg' , ...
        % % %             dagnn.Pooling('poolSize', [16 16], 'method', 'avg'), ...
        % % %             lastAdded.var, ...
        % % %             'prediction_avg') ;%12.11
        
        name =  'prediction_avg';
        net.addLayer('prediction_avg' , ...
            dagnn.Pooling('poolSize', [8 8], 'method', 'avg'), ...
            lastAdded.var, ...
            'prediction_avg') ;
        
        FC_vars = 128;
        lastAdded.var =name;
        
    case 2
        if strcmp(opts.o2p_method , 'MPN-COV') % 12.11
            name = 'mpn_cov_pool';
            net.addLayer(name , ...
                dagnn.MPN_COV_Pool_C('method', [],...
                'regu_method', 'power', ...
                'alpha', 0.5,...
                'epsilon', 0), ...
                lastAdded.var, ...
                {name, [name, '_aux_S'], [name, '_aux_V'],[name,'_aux_D']});
            lastAdded.var = name;
            FC_vars = 128*129/2;
        elseif strcmp(opts.o2p_method ,'MPN-COV-MATLAB')
            name = 'mpn_cov_pool';
            net.addLayer(name , ...
                dagnn.MPN_COV_Pool('method',  'covariance_small',   'regu_method', 'power',   'alpha', 0.5), ...
                lastAdded.var, ...
                {name, [name, '_aux_V'], [name, '_aux_S']}) ;
            lastAdded.var = name;
            FC_vars = 128*129/2;
            lastAdded.depth = 8256;%128
            
        elseif strcmp(opts.o2p_method , 'iSQRT')
            name = 'cov_pool';
            net.addLayer(name , dagnn.OBJ_ConvNet_COV_Pool(),           lastAdded.var,   name) ;
            lastAdded.var = name;
            
            name = 'cov_trace_norm';
            name_tr =  [name '_tr'];
            net.addLayer(name , dagnn.OBJ_ConvNet_Cov_TraceNorm(),   lastAdded.var,   {name, name_tr}) ;
            lastAdded.var = name;
            
            name = 'Cov_Sqrtm';
            net.addLayer(name , dagnn.OBJ_ConvNet_Cov_Sqrtm( 'coef', 1, 'iterNum', 5),    lastAdded.var,   {name, [name '_Y'], [name, '_Z']}) ;
            lastAdded.var = name;
            lastAdded.depth = lastAdded.depth * (lastAdded.depth + 1) / 2;
            
            name = 'Cov_ScaleTr';
            net.addLayer(name , dagnn.OBJ_ConvNet_COV_ScaleTr(),       {lastAdded.var, name_tr},  name) ;
            lastAdded.var = name;
            FC_vars = 128*129/2;
        elseif strcmp(opts.o2p_method , 'SVD-PN')
            name = 'SVD_PN' ;
            
            net.addLayer(name , dagnn.SVD_PN() ,lastAdded.var,...
                {name, [name, '_aux_U'], [name, '_aux_S'], [name, '_aux_V']}) ;

%             net.addLayer(name , dagnn.SVD_PN() ,lastAdded.var, name) ;
            
            lastAdded.var = name;
                
            FC_vars = 128*129/2;
            
            
        end
    otherwise
        error('valid order')
end

net.addLayer('prediction' , ...
    dagnn.Conv('size', [1 1 FC_vars  1000]), ...
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
%                                                           Meta parameters
% -------------------------------------------------------------------------

net.meta.normalization.imageSize = [64 64 3] ;
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;%?????????
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 64 ;%no scalejitter for ImageNet64
net.meta.normalization.averageImage = opts.averageImage ;

% net.meta.classes.name = opts.classNames ;
% net.meta.classes.description = opts.classDescriptions ;

% % net.meta.augmentation.jitterLocation = true ;
% % net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterBrightness = double(0.1 * opts.colorDeviation) ;

% % net.meta.augmentation.jitterAspect = [3/4, 4/3] ;
% % net.meta.augmentation.jitterScale  = [0.5, 1.1] ;
net.meta.augmentation.jitterSaturation = 0.2 ;
net.meta.augmentation.jitterContrast = 0.3 ;

net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;%???????????????

%lr = logspace(-1, -3, 60) ;

% lr =[0.1 * ones(1,8), 0.01*ones(1,5), 0.001*ones(1,6) , 1e-4*ones(1,6)] ;
%2order:batchsize:128*3  lr:1-->8 9-->13 *0.14
%1order:128

% lr = 1.5*[0.1 * ones(1,50), 0.01*ones(1,10), 0.001*ones(1,10) , 1e-4*ones(1,10) ,  1e-5*ones(1,10)] ;
lr = 3.5 *[0.1 * ones(1,50), 0.01*ones(1,15), 0.001*ones(1,15) ,...
    1e-4*ones(1,10) ,  1e-5*ones(1,10),1e-6*ones(1,10)] ;
% lr = logspace(-1,-4,60) ;

% lr =[0.1 * ones(1,10), 0.01*ones(1,10), 0.001*ones(1,10) , 1e-4*ones(1,10)] *0.1 ;

% lr =0.1*[0.1 * ones(1,10), 0.01*ones(1,10), 0.001*ones(1,10) , 1e-4*ones(1,10)] ;
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.batchSize = 128 * 2;
net.meta.trainOpts.numSubBatches = 1 ;
net.meta.trainOpts.weightDecay = 0.0001 ;

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
