function net = cnn_MoE_init_resnet(varargin)
%CNN_IMAGENET_INIT_RESNET  Initialize the ResNet-50 model for ImageNet classification

opts.order = 1;
opts.sectionLen = 4;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.o2p_method = [];
opts.comb_method = [] ;
opts.topk = [];
opts.branch_num = [];
opts.loss = [];
opts.loss_w = [];
opts.loss_comb_method = [];
opts.extra_loss = [];
opts.wx_method = 'wx';
opts.gating = [];
opts = vl_argparse(opts, varargin) ;
loss_wx = true;

net = dagnn.DagNN() ;

lastAdded.var = 'input' ;
lastAdded.depth = 3 ;
folder = ['ResNet18-1st'];

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


    function Conv_MoE(name, ksize, depth, varargin)
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
                dagnn.ReLU_MoE(), ...
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

sectionLen = opts.sectionLen ;  ;%9 : 74layers

for s = 2:5
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

branch_dim = 2^(s+3);
last_var =  lastAdded.var;

POOL_NAME = 'global_pool';
net.addLayer(POOL_NAME , ...
    dagnn.Pooling('poolSize', [8 8], 'method', 'avg'), ...
    lastAdded.var,POOL_NAME) ;
lastAdded.var = POOL_NAME ;
f_size = 1 ;%1st order

switch opts.wx_method
    case 'wx'
        wx_par = {'gating_w','gating_b'} ;
        folder = strcat(folder , '-NOnoise');
    case 'noise'
        wx_par = {'gating_w','gating_noise_w'} ;
        folder = strcat(folder , '-noise');
end
ksize = 1;
if strfind(opts.wx_method , 'FC')
    feature_var = lastAdded.var;
        net.addLayer('wx_g', ...
        dagnn.Conv('size', [ksize ksize 128 opts.branch_num], ...
        'stride', 1, ....
        'pad', (ksize - 1) / 2, ...
        'hasBias', false, ...
        'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
        feature_var, ...
        ['wx_g'], ...
        {'wx_g_f'} ) ;
    
         net.addLayer('wx_n', ...
        dagnn.Conv('size', [ksize ksize 128 opts.branch_num], ...
        'stride', 1, ....
        'pad', (ksize - 1) / 2, ...
        'hasBias', false, ...
        'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
        feature_var, ...
        ['wx_n'], ...
        {'wx_n_f'} ) ;
    
        net.addLayer('wx_sum' , ...
            dagnn.wx_Sum(), ...
            {'wx_g', 'wx_n'}, ...
            'wx') ;
        folder = strcat(folder , '-FCwx');
        
else
net.addLayer('wx' ,dagnn.wx('wx_method' , opts.wx_method ,...
    'size' , [ f_size*f_size*2^(s+2) opts.branch_num] , ...
    'branch' , opts.branch_num , 'topk',opts.topk , ...
    'f_size'  , [f_size f_size ] ,'ldim' , 2^(s + 2) ,...
    'hdim' , branch_dim ),...
    lastAdded.var,'wx',wx_par );
end

if ~isempty(opts.extra_loss)
    switch opts.extra_loss
        case 'importance'
            net.addLayer('loss_importance',dagnn.loss_importance_wx(...
                'loss_comb_method' , opts.loss_comb_method ,...
                'loss_weight',opts.loss_w ,'branch_num',...
                opts.branch_num), 'wx','loss_imp');
            net.vars(end).precious = true ;
            %             case 'loss_load'
            folder = strcat(folder , '-loss_imp');
        otherwise error('undefined loss case')
    end
else
    folder = strcat(folder , '-NO_loss');
end

net.addLayer('gating' , dagnn.gating('comb_method' , opts.comb_method ,...
    'size' , [ f_size*f_size*2^(s+2) opts.branch_num] , 'branch' , opts.branch_num ,...
    'topk',opts.topk , 'f_size'  , [f_size f_size ] ,...
    'ldim' , 2^(s + 2) ,...
    'hdim' , branch_dim ),'wx','gating');

folder = strcat(folder , '-gating_', opts.comb_method);

%branch~experts
for b = 1:opts.branch_num
    lastAdded.var = POOL_NAME;
    name =  ['branch_' num2str(b)];
    Conv(name , 1, branch_dim ) ;
%     name =  [name '_pooling'];
%     % %     net.addLayer(name , ...
%     % %         dagnn.Pooling_MoE('poolSize', [f_size f_size], 'method', 'avg'), ...
%     % %         lastAdded.var,  name) ;%2018.3.5 19:50
%     net.addLayer(name , ...
%         dagnn.Pooling('poolSize', [f_size f_size], 'method', 'avg'), ...
%         lastAdded.var,name) ;
    
    lastAdded.depth = 2^(s+2) ;
end

% % %branch_out
in_{1} = 'gating';
for ee = 1:opts.branch_num
%     in_{ee + 1} = ['branch_', num2str(ee) , '_pooling'];
    in_{ee + 1} = ['branch_', num2str(ee) , '_relu'];
end
net.addLayer( 'branch_out' , ...
    dagnn.branch_out('branch_num' ,opts.branch_num ,'f_size'  , [f_size f_size] ,...
    'ldim' , 2^(s+2), 'hdim' , branch_dim  ,'topk' ,opts.topk),...
    in_ , 'branch_out' ) ;
lastAdded.var = 'branch_out' ;
% FC_vars = branch_dim  ;%%gzl 2018.3.4
FC_vars = branch_dim * opts.topk ;%

folder = strcat(folder , '-expert_' , num2str(opts.branch_num));

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
lr = 1 *[0.1 * ones(1,50), 0.01*ones(1,15), 0.001*ones(1,15) ,...
    1e-4*ones(1,10) ,  1e-5*ones(1,11)] ;
% lr = logspace(-1,-4,60) ;
folder = strcat(folder , '-LR_',num2str(lr(1)));
% lr =[0.1 * ones(1,10), 0.01*ones(1,10), 0.001*ones(1,10) , 1e-4*ones(1,10)] *0.1 ;

% lr =0.1*[0.1 * ones(1,10), 0.01*ones(1,10), 0.001*ones(1,10) , 1e-4*ones(1,10)] ;
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.batchSize = 128 * 2;
% net.meta.trainOpts.batchSize = 20;
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
