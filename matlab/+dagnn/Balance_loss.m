classdef Balance_loss < dagnn.ElementWise
%Extra loss to ensure balance selection in one mini-batch
%created by Zilin Gao
    properties
        loss_weight = 100;
        CM_num = 16 ;
    end
    
    methods
        
        function forwardAdvanced(obj, layer)            
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            input = net.vars(in).value;
            if iscell(input)
                input = input{1};
            end   
            
            w_sum = sum(input,2);
            net.vars(out).value{3} = w_sum;
            RSD = std(w_sum) / mean(w_sum) ;%relative standard deviation
            net.vars(out).value{1} = obj.loss_weight * RSD^2 ;
            net.vars(out).value{2} = RSD ;
            
            assert(~any(isnan( net.vars(out).value{1}(:))))
            net.numPendingVarRefs(in) = net.numPendingVarRefs(in) - 1;
            % clear inputs if not needed anymore
            if net.numPendingVarRefs(in) == 0
                if ~net.vars(in).precious & ~net.computingDerivative & net.conserveMemory
                    net.vars(in).value = [] ;
                end
            end                        
        end
        
        function backwardAdvanced(obj, layer)
            if  ~obj.net.conserveMemory
                backwardAdvanced@dagnn.Layer(obj, layer) ;
                return ;
            end
            net = obj.net ;
            in = layer.inputIndexes ;
            out = layer.outputIndexes ;
            
            w_sum = net.vars(out).value{3};
            m = mean(w_sum);
            N = obj.CM_num;
            w = obj.loss_weight;
            RSD = net.vars(out).value{2};
            in_data = net.vars(in).value;
            if iscell(in_data)
                in_data = in_data{1};
            end
            
            der_sfm = w_sum * 2 * w  / ((N-1) * m^2) - ...
                2* w  * ( 1 / (N-1) + RSD ^2 /N) / m;
            derInput = der_sfm * ones(1,size(in_data ,2 ));
            
            if ~net.vars(out).precious & net.conserveMemory
                net.vars(out).value = [] ;
            end
            
            if net.numPendingVarRefs(in) == 0
                net.vars(in).der = derInput ;
            else
                net.vars(in).der = net.vars(in).der + derInput ;
            end
            net.numPendingVarRefs(in) = net.numPendingVarRefs(in) + 1 ;
            assert(~any(isnan(net.vars(in).der(:))))
        end
        
        function obj = Balance_loss(varargin)
            obj.load(varargin) ;
        end
    end
end
