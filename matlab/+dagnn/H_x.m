classdef H_x < dagnn.Layer
    properties
        size = [];
        hasBias = false;
        topk = 8;
        CM_num = 16;
        f_size = [1 1] ;
        dim_in = 128;
        
    end
    
    methods
        
        function forwardAdvanced(obj, layer)
            %H_x = wg * x + gamma_ * log(1 + exp(wn * x))
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            param = layer.paramIndexes;
            bs = size(net.vars(in).value , 4) ;
            input_data = reshape(net.vars(in).value , [] , bs);
            
            wgx = net.params(param(1)).value' * input_data;
            
            if strcmp(net.mode,'test')
                gamma_ = zeros(1,bs);
            else
%                 gamma_ = unifrnd(-1,1,1,bs);
                gamma_ = random('Normal',0,1,1,bs);%normal distribution for each img
            end
            wnx = net.params(param(2)).value' * input_data ;
            sp = log(1 + exp(wnx));%softplus
            if any(isinf(sp(:)))
                inf_n = isinf(sp);
                sp(inf_n) = wnx(inf_n);
            end
            noise = bsxfun(@times ,sp , gamma_) ;
            
            H_x = wgx + noise;
            net.vars(out).value{2}  = gamma_ ;
            
            net.vars(out).value{1}  = H_x ;
            assert(~any(isnan(H_x(:))))
            assert(~any(isinf(H_x(:))))
            net.numPendingVarRefs(in) = net.numPendingVarRefs(in) - 1;
            
            % clear inputs if not needed anymore
            if net.numPendingVarRefs(in) == 0
                if ~net.vars(in).precious & ~net.computingDerivative & net.conserveMemory
                    net.vars(in).value = [] ;
                end
            end
            
        end
        
        function backwardAdvanced(obj, layer)
            %H_x = wg * x + gamma_ * log(1 + exp(wn * x))           
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            param = layer.paramIndexes;
            
            if isempty(net.vars(out).der), return ; end
            
            dldtopk = net.vars(out).der;
            bs = size(dldtopk , 2);
            x = net.vars(in).value ;
            x = reshape(x , [], bs);
            dldx = net.params(param(1)).value * dldtopk ;
            dldwg = x * dldtopk' ;
            
            wn = net.params(param(2)).value ;
            s = net.vars(out).value{2};
            e = exp(wn' * x) ;
            dldwntx = bsxfun(@times , dldtopk .* (1 -  1./ ( 1+e ) ), s );
            dldx2 = wn * dldwntx ;
            dldx = dldx + dldx2 ;
            
            dldwn = x * dldwntx' ;
            net.params(param(2)).der = dldwn;
            
            derInput = reshape(dldx , obj.f_size(1),obj.f_size(1), obj.dim_in  ,[]);
            net.params(param(1)).der = dldwg;
            assert(~any(isnan(dldwn(:))))
            assert(~any(isnan(dldwg(:))))
            
            if ~net.vars(out).precious  & net.conserveMemory
                net.vars(out).der = [] ;
                net.vars(out).value = [] ;
            end
            
            if net.numPendingVarRefs(in) == 0
                net.vars(in).der = derInput ;
            else
                net.vars(in).der = net.vars(in).der + derInput ;
            end
            net.numPendingVarRefs(in) = net.numPendingVarRefs(in) + 1 ;
        end
        
        function params = initParams(obj)
            
            sc = sqrt(2 / obj.size(1)) ;
            %for balance select in first several mini-batches, ones initial is
            %employed
            params{1} = ones(obj.size,'single') / obj.size(1);%balance initial
            params{2} = ones(obj.size,'single')/ obj.size(1);
        end
        
        function obj = H_x(varargin)
            obj.load(varargin) ;
        end
    end
end
