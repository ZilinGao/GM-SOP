classdef OBJ_ConvNet_COV_Pool < dagnn.Layer
  properties
    opts = {'cuDNN'}
    cudnn = {'CuDNN'} 
  end


  
  methods
    function outputs = forward(self, inputs, params)
        %[outputs{1}] =  ConvNet_Cov_Pool(inputs{1});  % lph 2017/10/14 11:37
        
        % %res(i+1).x    =  vl_nncov_pool(res(i).x,   cudnn{:});
               [outputs{1}] =  vl_nncov_pool(inputs{1}, self.cudnn{:});
    end
   
     function [derInputs, derParams] = backward(self, inputs, params, derOutputs, outputs)
         % [~, derInputs{1}]=  ConvNet_Cov_Pool(inputs{1}, derOutputs{1}); % lph 2017/10/14 11:37
         
         %   % res(i).dzdx      = vl_nncov_pool(res(i).x,    res(i+1).dzdx,  cudnn{:});
                   [derInputs{1}] = vl_nncov_pool(inputs{1}, derOutputs{1}, self.cudnn{:});
         derParams  = {} ;
    end
    
    function obj = OBJ_ConvNet_COV_Pool(varargin)
      obj.load(varargin) ;
    end
  end
end
