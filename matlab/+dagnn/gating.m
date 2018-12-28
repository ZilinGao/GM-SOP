classdef gating < dagnn.Layer
    properties
        size = [];
        hasBias = false;
        topk = 8;
        CM_num = 16;
        f_size = [8 8] ;
    end
    
    methods
               
        function forwardAdvanced(obj, layer)
            %select topk CM and normalize weight by softmax
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            H_x = net.vars(in).value;
            if iscell(H_x)
                H_x = H_x{1};
            end
            if numel(size(H_x)) ~= 2
                H_x = squeeze(H_x);
            end
            
            q = sort(H_x, 'descend' );
            CM_index = H_x >= (ones(obj.CM_num,1) * q(obj.topk , :));
            
            if ~any(sum(CM_index) ~= obj.topk)
                ;%do nothing
            elseif sum(sum(CM_index,1),2) == numel(CM_index)
                %random select topk if all weights are equal 
                %(1st batch with ones initial w)
                if obj.topk ~= obj.CM_num
                    t_rand = randn(size(CM_index));%generate a fake weight matrix randomly
                    q_rand = sort(t_rand , 'descend');
                    CM_index = t_rand >= (ones(obj.CM_num,1) * q_rand(obj.topk, :));
                end
            else
                mm = find(sum(CM_index) ~= obj.topk);
                for m1 = 1:numel(mm)
                    more_ = find(H_x(:,mm(m1)) == q(obj.topk, mm(m1)));
                    discart_num = sum(CM_index(:,mm(m1))) - obj.topk;
                    discart_id = more_(end - discart_num +1 :end);
                    CM_index(discart_id,mm(m1)) = false;
                end
            end
            
            %topk
            topk_v = CM_index .* H_x;
            net.vars(out).value{3}  = topk_v;
            
            %softmax
            topk_v(topk_v == 0) =-inf;
            %minus max as matconvnet
            mm = max(topk_v);
            topk_v = bsxfun(@minus , topk_v , mm);
            sfm = bsxfun(@rdivide, exp(topk_v) , sum(exp(topk_v)));
            
            assert(gather(~any(isnan(sfm(:)))) == 1)
            net.vars(out).value{1} = sfm ;
            net.vars(out).value{2} = CM_index;
%             if numel(net.vars(out).value )< 6
%                 net.vars(out).value{6} = zeros(1,obj.CM_num);
%             end
%             net.vars(out).value{6} = sum(sfm') + net.vars(out).value{6};

             net.vars(out).value{6} = sum(sfm');
            
            net.numPendingVarRefs(in) = net.numPendingVarRefs(in) - 1;
            % clear inputs if not needed anymore
            if net.numPendingVarRefs(in) == 0
                if ~net.vars(in).precious & ~net.computingDerivative & net.conserveMemory
                    net.vars(in).value = [] ;
                end
            end
            
        end
        
        function backwardAdvanced(obj, layer)
            
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            param = layer.paramIndexes;
            
            if isempty(net.vars(out).der), return ; end
            
            in_data = net.vars(in).value ;
            if iscell(in_data)
                in_data = in_data{1} ;
            end
            if numel(size(in_data)) ~= 2
                in_data = squeeze(in_data);
                tensor_flag = true;
            else
                tensor_flag = false;
            end
            
            y = net.vars(out).value{1};
            dldsfm = net.vars(out).der;
            dldtopk = y .* bsxfun(@minus , dldsfm , sum(dldsfm .* y));
            
            derInput = dldtopk;%wx + b
            assert(gather(~any(isnan(derInput(:)))) ==1)
            
            if ~net.vars(out).precious & net.conserveMemory
                net.vars(out).der = [] ;
                net.vars(out).value = [] ;
            end
            if tensor_flag
                derInput = reshape(derInput , 1,1,obj.CM_num,[]);
            end
            if net.numPendingVarRefs(in) == 0
                net.vars(in).der = derInput ;
            else
                net.vars(in).der = net.vars(in).der + derInput ;
            end
            assert(gather(~any(isnan(net.vars(in).der(:)))) == 1)
            net.numPendingVarRefs(in) = net.numPendingVarRefs(in) + 1 ;
        end
        
        
        function obj = gating(varargin)
            obj.load(varargin) ;
        end
    end
end
