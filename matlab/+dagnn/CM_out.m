classdef CM_out < dagnn.Layer
    properties
        CM_num = 16;
        topk = 8;
        hdim = 256 ;
    end
    
    methods
        
        function forwardAdvanced(obj, layer)
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            bs = size(net.vars(in(1)).value{1},2);
            
            net = topk_forward(obj,net,in,out,bs);
            
            % clear inputs if not needed anymore
            for v = in
                net.numPendingVarRefs(v) = net.numPendingVarRefs(v) - 1 ;
                if net.numPendingVarRefs(v) == 0
                    if ~net.vars(v).precious & ~net.computingDerivative & net.conserveMemory
                        net.vars(v).value = [] ;
                    end
                end
            end
        end
        
        function backwardAdvanced(obj, layer)
            
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            
            if isempty(net.vars(out).der), return ; end
            derOutput = net.vars(out).der ; %1*1*(dim*CM_num)*BS
            bs = size(derOutput , 4);
            
            net = topk_backward(obj, net, in, out, bs, derOutput);
            
            if ~any(net.vars(out).precious) & net.conserveMemory
                net.vars(out).der = [] ;
                net.vars(out).value = [] ;
            end
        end
        
        function net = topk_forward(obj, net, in, out, bs)
            %in(1):weights   in(2:end):outputs of CMs
            fusion_input = squeeze(cat(4,net.vars(in(2:end)).value));
           %[dim1_cm1_sample1 ...  dim1_cm1_samplek   dim1_cm2_sample1... dim1_cm2_samplej...dim1_cmN_sampleq;
           % dim2_cm1_sample1 ...  dim2_cm1_samplek   dim2_cm2_sample1... dim2_cm2_samplej...dim2_cmN_sampleq;
           % ...
           % dimD_cm1_sample1 ...  dimD_cm1_samplek   dimN_cm2_sample1... dimD_cm2_samplej...dimD_cmN_sampleq]
            
            index = net.vars(in(1)).value{2};
            MAP_CM = cumsum(index,2) .* index ;%(i,k):img k is the nth sample in CM i
            samples = sum(index,2);%number of samples received by each CM
            START = cumsum([0; samples(1:end-1)]);
            id =bsxfun(@plus, MAP_CM, START) .* index;%the NO. in all images
            weight =net.vars(in(1)).value{1};
            weight(index == 0) = [];%score: topk * bs
            
            weight = reshape(weight,[],1);
            id = reshape(id,[],1);  id(id==0)=[];
            net.vars(out).value = bsxfun(@times, fusion_input(:,id') , weight');%hdim*(topk*bs) 
            %fusion_input(:,id'):
           %[dim1_cm1_sample1 ... dim1_cmk_sample1   dim1_cm1_sample2 ... dim1_cmk_sample2 ... dim1_cmN_sampleq;
           % dim2_cm1_sample1 ... dim2_cmk_sample1   dim2_cm1_sample2 ... dim2_cmk_sample2 ... dim2_cmN_sampleq;
           % ...
           % dimD_cm1_sample1 ... dimD_cmk_sample1   dimN_cm1_sample2 ... dimD_cmk_sample2 ... dimD_cmN_sampleq]
           
           % weight'
           %[cm1_sample1     ...    cmk_sample1       cm1_sample2     ...     cmk_sample2  ...  cmN_sampleq]
           
           %cmi_samplej indicates the i-th selected CM for sample j
            
            net.vars(out).value = reshape(net.vars(out).value,1,1,obj.hdim, obj.topk, bs);
            net.vars(out).value = reshape(sum(net.vars(out).value, 4), 1, 1, obj.hdim, bs);
            
        end
        
        
        function net = topk_backward(obj, net, in, out, bs, derOutput)
            derOutput = reshape(derOutput , obj.hdim,[]);     %dim * bs
            index = net.vars(in(1)).value{2} ;
            weight = net.vars(in(1)).value{1};%CM_num * bs
            weight(index == 0) = [];% topk * bs            
            der_ = repmat(derOutput, 1, 1, obj.topk);
            der_ = permute(der_,[1 3 2]);%dim * topk * bs
            derOutput = reshape(der_,obj.hdim, []);
            clear der_
            
            dldximg = bsxfun(@times, derOutput , weight );

           %derOutput   dim * (topk * bs)
           %[dim1_cm1_sample1 ... dim1_cmk_sample1  dim1_cm1_sample2 ... dim1_cmk_sample2 ... dim1_cmk_sampleq; 
           % dim2_cm1_sample1 ... dim2_cmk_sample1  dim2_cm1_sample2 ... dim2_cmk_sample2 ... dim2_cmk_sampleq; 
           % ...
           % dimD_cm1_sample1 ... dimD_cmk_sample1  dimD_cm1_sample2 ... dimD_cmk_sample2 ... dimD_cmk_sampleq] 
           
           % weight   1 * (topk * bs£©
           %[cm1_sample1     ...    cmk_sample1       cm1_sample2     ...cmk_sample2  ...  cmN_sampleq]
           
            %derivative of each CM
            MAP = cumsum(index) .* index ;%CM i is the p-th selected CM in sample i, p=id(i,k)
            START_e = [0 cumsum(sum(index))];START_e(end) = [];
            
            id_e =bsxfun(@plus, MAP , START_e) .* index;%the NO. in all samples
            id_e = reshape(id_e', [], 1)';  id_e(id_e==0)=[];
            samples = sum(index,2);
            
            dldx = reshape(dldximg(:,id_e),1,1,obj.hdim,[]);%arrange follow CM
            
            for ee = 1:obj.CM_num
                if samples(ee) == 0
                    continue%numPendingVarRefs not statistic
                end
                if net.numPendingVarRefs(in(ee+1)) == 0
                    net.vars(in(ee +1 )).der = dldx(:,:,:,1:samples(ee));
                else
                    net.vars(in(ee+1)).der = net.vars(in(ee+1)).der + dldx(:,:,:,1:samples(ee));
                end
                dldx(:,:,:,1:samples(ee)) = [];
                assert(gather(~any(isnan(net.vars(in(ee+1)).der(:))))==1)
                net.numPendingVarRefs(in(ee+1)) = net.numPendingVarRefs(in(ee+1)) + 1 ;
            end
            
            %derivative of gating module
            MAP_CM = cumsum(index,2) .* index ;%(i,k): sample k is the q-th sample received in CM i
            START = cumsum([0; samples(1:end-1)]);
            id =bsxfun(@plus, MAP_CM , START) .* index;%the NO in all samples
            id = reshape(id,[], 1);id(id==0) = [];
            x = cat(4,net.vars(in(2:end)).value);%CM1_sample1,CM1_samplej...CM2_samplen.
            x = x(:,:,:,id);%ablation the feature did not be selected 
            dldsfm = sum(reshape(x,obj.hdim,[]) .* derOutput);
            DLDSFM_in = zeros(obj.CM_num, bs,'single');
            if strcmp(net.device,'gpu')
                DLDSFM_in = gpuArray(DLDSFM_in);
            end
            DLDSFM_in(index) = dldsfm;
            
            if net.numPendingVarRefs(in(1)) == 0
                net.vars(in(1)).der = DLDSFM_in;
            else
                net.vars(in(1)).der = DLDSFM_in + net.vars(in(1)).der;
            end
            assert(gather(~any(isnan(net.vars(in(1)).der(:))))==1)
            net.numPendingVarRefs(in(1)) = net.numPendingVarRefs(in(1)) + 1 ;
        end
        
        
        function obj = CM_out(varargin)
            obj.load(varargin) ;
        end
    end
end
