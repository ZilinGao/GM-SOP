function net = cnn_MoE_init_resnet_gating_isqrt_hx(varargin)
%CNN_IMAGENET_INIT_RESNET  Initialize the ResNet-50 model for ImageNet classification

opts.order = 2;
opts.sectionLen = 6;
opts.averageImage = zeros(3,1) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.topk = [];
opts.branch_num = [];
opts.loss_w = [];
opts.cat_bn= true ;
opts.agg_drop = true;
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

lastAdded.var = 'input' ;
lastAdded.depth = 3 ;
folder = ['ResNet50-2nd'];

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
        %         net.addLayer([name '_bn'], ...
        %             dagnn.BatchNorm('numChannels', depth), ...
        %             [name '_conv'], ...
        %             [name '_bn'], ...
        %             {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
        %         lastAdded.depth = depth ;
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
        if l == 1 & s>2
            % %    if l == 1
            Conv([name '_adapt_conv'], 1, 2^(s+2), ...
                'downsample', s <5 | (s == 5 & opts.order == 1), 'relu', false) ;
        end
        sumInput = lastAdded ;
        
        % AB: 3x3, 3x3: downsample if first segment in section from
        % section 3 onwards.
        lastAdded = sectionInput ;
        Conv([name 'a'], 3, 2^(s+2), ...
            'downsample', (s == 3 | s==4 |(s == 5 & opts.order == 1)) & l == 1) ;
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

branch_dim = 64; % dim of last conv in each component model
last_var =  lastAdded.var;

g_conv_size = 16 ; %size of feature inputted to 2nd order layer
gating_dim1 = 128; %dim of 1st conv in gating
gating_dim2 = 64; %dim of 2nd conv in gating
gating_dim3 = gating_dim2 * (gating_dim2 + 1)/2; %dim of 2nd order feature in gating module

name = 'gating';
Conv([name '1'], 1, gating_dim1, 'relu', true) ;
Conv([name '2'], 1, gating_dim2, 'relu', true) ;

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

wx_par = {'gating_w','gating_noise_w'} ;

net.addLayer('wx' ,dagnn.wx( ...
    'size' , [ 1*1*gating_dim2*(gating_dim2+1)/2 opts.branch_num] , ...
    'branch' , opts.branch_num , 'topk',opts.topk , ...
    'f_size'  , [1 1 ] ,'ldim' , gating_dim3 ,...
    'hdim' , branch_dim ),...
    lastAdded.var,'wx',wx_par );

net.addLayer('gating' , dagnn.gating( ...
    'size' , [ 1*1*gating_dim3 opts.branch_num] , ...
    'branch' , opts.branch_num ,...
    'topk',opts.topk , 'f_size'  , [1 1 ] ,...
    'ldim' , gating_dim3 ,...
    'hdim' , branch_dim ),...
    'wx','gating');

loss_input = 'gating';

net.addLayer('loss_importance',dagnn.loss_importance_wx(...
    'loss_weight',opts.loss_w ,'branch_num',...
    opts.branch_num),...
    loss_input,'loss_imp');
net.vars(end).precious = true ;



% -------------------------------------------------------------------------
% Component Module
% -------------------------------------------------------------------------

lastAdded.depth = 2^(s+2);
mid_dim = 96; %dim of 2nd conv  in each CM

for b = 1:opts.branch_num
    lastAdded.var =  last_var;
    br_name =  ['branch_' num2str(b)];
    
    Conv_nobn([br_name '_a'] , 1, mid_dim) ;
    lastAdded.depth = mid_dim;
    
    Conv_nobn([br_name '_b'] , 1, branch_dim ) ;
    
    br_name = [br_name '_'];
    name = [br_name 'cov_pool'];
    net.addLayer(name , dagnn.OBJ_ConvNet_COV_Pool(), ...
        lastAdded.var,   name) ;
    lastAdded.var = name;
    
    name = [br_name 'cov_trace_norm'];
    name_tr =  [name '_tr'];
    net.addLayer(name , dagnn.OBJ_ConvNet_Cov_TraceNorm(),  ...
        lastAdded.var,   {name, name_tr}) ;
    lastAdded.var = name;
    
    name = [br_name 'Cov_Sqrtm'];
    net.addLayer(name , dagnn.OBJ_ConvNet_Cov_Sqrtm( 'coef', 1, 'iterNum', 3),  ...
        lastAdded.var,   {name, [name '_Y'], [name, '_Z']}) ;
    lastAdded.var = name;
    lastAdded.depth = lastAdded.depth * (lastAdded.depth + 1) / 2;
    
    name = [br_name 'Cov_ScaleTr'];
    net.addLayer(name , dagnn.OBJ_ConvNet_COV_ScaleTr(),  ...
        {lastAdded.var, name_tr},  name) ;
        
    lastAdded.depth = 2^(s+2) ;
end



% -------------------------------------------------------------------------
% Aggregation
% -------------------------------------------------------------------------

in_{1} = 'gating'; %inputs of aggregation layer
for ee = 1:opts.branch_num  
    in_{ee + 1} = ['branch_', num2str(ee) , '_Cov_ScaleTr'];   
end

net.addLayer( 'branch_out' , ...
    dagnn.branch_out('branch_num' ,opts.branch_num ,...
    'f_size'  , [g_conv_size g_conv_size] ,...
    'ldim' , 2^(s+2), 'hdim' , branch_dim * (branch_dim + 1) /2  ,...
    'topk' ,opts.topk),...
    in_ , 'branch_out' ) ;
lastAdded.var = 'branch_out' ;

FC_vars = branch_dim * (branch_dim + 1) /2 ;%final representation dim
lastAdded.depth = branch_dim * (branch_dim + 1) /2;

folder = strcat(folder , '-CM_' , num2str(opts.topk));
if opts.branch_num ~=opts.topk
    folder = strcat(folder,'from', num2str(opts.branch_num));
end

if opts.cat_bn  %add a bn layer after aggregation operation
    name = 'cat_bn';
    net.addLayer(name, ...
        dagnn.BatchNorm('numChannels', lastAdded.depth), ...
        lastAdded.var, ...
        [name '_bn'], ...
        {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
    lastAdded.var = [name '_bn'] ;    
    
end

if opts.agg_drop %add a dropout layer after bn in case of overfitting
    name = ['agg_drop'];
    drop_rate = 0.2;
    net.addLayer(name,dagnn.DropOut('rate',drop_rate),...
        lastAdded.var,name);
    lastAdded.var = name;
end


% -------------------------------------------------------------------------
% Inference Layers
% -------------------------------------------------------------------------

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
%                                                           Meta
% -------------------------------------------------------------------------

net.meta.normalization.imageSize = [64 64 3] ;%for imagenet64 dataset
net.meta.inputSize = [net.meta.normalization.imageSize, 32] ;
net.meta.normalization.averageImage = opts.averageImage ;
net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;

lr = 1 *[0.1 * ones(1,50), 0.01*ones(1,10), 0.001*ones(1,5) , 1e-4*ones(1,5)] ;
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
% adjust calculation order for this multi-branch model
if ~isnan(net.getLayerIndex('gating'))
    net.ExecutionOrder_experts();
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
